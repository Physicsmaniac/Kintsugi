from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from PIL import Image, ImageOps
from datasets import load_dataset

from src.models.page_embedder import PageEmbeddingNet, SupConLoss
from src.data.preprocessing import preprocess_single_strip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP probe logs from HuggingFace/httpx internals
for _noisy in ("httpx", "datasets.load", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


class MultiPageStripDataset(IterableDataset):
    """Generates training batches for contrastive page embedding learning.
    
    For each batch:
    1. Sample P pages from the buffer (e.g., 8 different images = 8 "pages")
    2. From each page, shred into strips and sample S strips (e.g., 4)
    3. Yield (strip_image_tensor, page_label) pairs
    
    Batch structure: P x S strips with P distinct page labels.
    Uses chainyo/rvl-cdip streamed from HuggingFace. Each image is treated
    as a separate "page" (different documents = different pages).
    """
    
    def __init__(
        self,
        split: str = "train",
        pages_per_batch: int = 8,
        strips_per_page: int = 4,
        buffer_size: int = 500,
        num_strips_to_shred: int = 15,
        max_image_width: int = 400,
        dataset_path: str = "chainyo/rvl-cdip",
    ):
        super().__init__()
        self.split = split
        self.pages_per_batch = pages_per_batch
        self.strips_per_page = strips_per_page
        self.buffer_size = buffer_size
        self.num_strips_to_shred = num_strips_to_shred
        self.max_image_width = max_image_width
        self.dataset_path = dataset_path
        
        self.dataset = load_dataset(dataset_path, split=split, streaming=True)

    def _process_image(self, img: Image.Image) -> list[Image.Image]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w > self.max_image_width:
            new_h = int(h * (self.max_image_width / w))
            img = img.resize((self.max_image_width, new_h), Image.Resampling.LANCZOS)
            w, h = img.size
            
        strip_width = max(32, w // self.num_strips_to_shred)
        
        strips = []
        for i in range(0, w, strip_width):
            if i + strip_width > w:
                break
            box = (i, 0, i + strip_width, h)
            strip_img = img.crop(box)
            strips.append(strip_img)
            
        return strips

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset_stream = self.dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        else:
            dataset_stream = self.dataset

        buffer: list[Image.Image] = []
        page_label_counter = 0
        
        for item in dataset_stream:
            img = item["image"]
            buffer.append(img)
            
            # Warm up: only wait for 20 pages instead of full 500 to start yielding quickly
            if len(buffer) >= min(20, self.buffer_size):
                # Process a batch
                sampled_pages = random.sample(buffer, self.pages_per_batch)
                
                for page_img in sampled_pages:
                    all_strips = self._process_image(page_img)
                    
                    if len(all_strips) < self.strips_per_page:
                        continue
                        
                    sampled_strips = random.sample(all_strips, self.strips_per_page)
                    
                    for strip in sampled_strips:
                        # Resize to 32px width, pad to 224x224, normalize
                        strip = strip.resize((32, strip.height * 32 // strip.width), Image.Resampling.LANCZOS)
                        # Pad to 224x224
                        pad_w = max(0, 224 - strip.width)
                        pad_h = max(0, 224 - strip.height)
                        pad_left = pad_w // 2
                        pad_right = pad_w - pad_left
                        pad_top = pad_h // 2
                        pad_bottom = pad_h - pad_top
                        
                        padded_strip = ImageOps.expand(strip, border=(pad_left, pad_top, pad_right, pad_bottom), fill="white")
                        if padded_strip.size != (224, 224):
                            padded_strip = padded_strip.resize((224, 224))
                            
                        # Convert to tensor and normalize (mock normalization)
                        import torchvision.transforms.functional as TF
                        tensor = TF.to_tensor(padded_strip)
                        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        
                        yield tensor, page_label_counter
                    
                    page_label_counter += 1
                
                # Remove some images from buffer to allow new ones
                buffer = buffer[self.pages_per_batch:]


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, num_batches: int = 10) -> tuple[float, float]:
    """Compute average intra-page vs inter-page cosine similarity."""
    model.eval()
    intra_sims = []
    inter_sims = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            sim_matrix = torch.matmul(embeddings, embeddings.T)
            
            label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
            
            # Mask out diagonal (self similarity)
            eye = torch.eye(labels.size(0), device=device, dtype=torch.bool)
            
            intra_mask = label_matrix & ~eye
            inter_mask = ~label_matrix
            
            if intra_mask.any():
                intra_sims.append(sim_matrix[intra_mask].mean().item())
            if inter_mask.any():
                inter_sims.append(sim_matrix[inter_mask].mean().item())

    avg_intra = sum(intra_sims) / len(intra_sims) if intra_sims else 0.0
    avg_inter = sum(inter_sims) / len(inter_sims) if inter_sims else 0.0
    return avg_intra, avg_inter


def main() -> None:
    import time
    parser = argparse.ArgumentParser(description="Train PageEmbeddingNet")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--pages-per-batch", type=int, default=8, help="Pages per batch")
    parser.add_argument("--strips-per-page", type=int, default=4, help="Strips per page")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--train-steps-per-epoch", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--log-interval", type=int, default=50, help="Log progress every N steps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")

    # Dataset and DataLoader
    batch_size = args.pages_per_batch * args.strips_per_page
    train_dataset = MultiPageStripDataset(
        split="train", 
        pages_per_batch=args.pages_per_batch, 
        strips_per_page=args.strips_per_page
    )
    val_dataset = MultiPageStripDataset(
        split="val", 
        pages_per_batch=args.pages_per_batch, 
        strips_per_page=args.strips_per_page
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, Loss, Optimizer
    model = PageEmbeddingNet(embedding_dim=128).to(device)
    criterion = SupConLoss(temperature=0.07).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.amp.GradScaler(device.type)

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        logger.info(f"🔄 Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint.get("best_loss", best_loss)

    csv_path = Path(args.output_dir) / "training_log.csv"
    file_exists = csv_path.exists()
    with open(csv_path, mode="a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_intra_sim", "val_inter_sim", "sim_gap"])

    logger.info("🔥 Starting training loop...")
    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        logger.info("════════════════════════════════════════════════════════════════")
        logger.info(f"🏋️  Embedder Epoch {epoch + 1} / {args.epochs}  [lr: {current_lr:.2e}]")
        logger.info("════════════════════════════════════════════════════════════════")
        
        model.train()
        total_loss = 0.0
        train_iter = iter(train_loader)
        
        epoch_start_time = time.time()
        step_start_time = time.time()
        last_log_step = 0
        
        for step in range(1, args.train_steps_per_epoch + 1):
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)
                
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device.type):
                embeddings = model(images)
                loss = criterion(embeddings, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if step % args.log_interval == 0 or step == args.train_steps_per_epoch:
                elapsed = time.time() - step_start_time
                steps_passed = step - last_log_step
                samples_per_sec = (steps_passed * batch_size) / max(1e-5, elapsed)
                running_loss = total_loss / step
                logger.info(
                    f"   📈 [Embedder Step {step:4d}/{args.train_steps_per_epoch:4d}]  "
                    f"batch_loss: {loss.item():.4f} | run_loss: {running_loss:.4f} | {samples_per_sec:.1f} samples/sec"
                )
                step_start_time = time.time()
                last_log_step = step
                
        train_time = time.time() - epoch_start_time
        avg_train_loss = total_loss / args.train_steps_per_epoch
        scheduler.step()
        
        logger.info("🔍 Running validation embeddings assessment...")
        val_start_time = time.time()
        intra_sim, inter_sim = validate(model, val_loader, device)
        val_time = time.time() - val_start_time
        sim_gap = intra_sim - inter_sim

        logger.info(f"📊 Embedder Epoch {epoch + 1} Summary (Train: {train_time:.1f}s, Val: {val_time:.1f}s):")
        logger.info(f"   ├── Avg Train Loss:  {avg_train_loss:.4f}")
        logger.info(f"   ├── Intra-page Sim:  {intra_sim:.4f} (Higher is better)")
        logger.info(f"   ├── Inter-page Sim:  {inter_sim:.4f} (Lower is better)")
        logger.info(f"   └── Separation Gap:  {sim_gap:+.4f}")

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, intra_sim, inter_sim, sim_gap])
            
        ckpt_payload = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }

        # Save latest model
        latest_path = Path(args.output_dir) / "latest_checkpoint.pth"
        torch.save(ckpt_payload, latest_path)
        logger.info(f"💾 Saved latest checkpoint → {latest_path}")

        # Save best model by loss
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            ckpt_payload["best_loss"] = best_loss
            best_path = Path(args.output_dir) / "best_page_embedder.pth"
            torch.save(ckpt_payload, best_path)
            logger.info(f"🏆 New best loss ({best_loss:.4f})! Saved model → {best_path}")

if __name__ == "__main__":
    main()
