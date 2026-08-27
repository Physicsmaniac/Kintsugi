from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import signal
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms as T
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
for _noisy in ("httpx", "huggingface_hub"):
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
        streaming: bool = True,
    ):
        super().__init__()
        self.split = split
        self.pages_per_batch = pages_per_batch
        self.strips_per_page = strips_per_page
        self.buffer_size = max(pages_per_batch * 2, buffer_size)
        self.num_strips_to_shred = num_strips_to_shred
        self.max_image_width = max(64, max_image_width)
        self.dataset_path = dataset_path
        self.streaming = streaming
        self._local_ds = None

        # Load local datasets once in the main process to avoid file lock
        # deadlocks when multiple PyTorch workers spawn simultaneously.
        if not self.streaming:
            self._local_ds = load_dataset(self.dataset_path, split=self.split, streaming=False)

        self.strip_augment = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.5),
            T.RandomGrayscale(p=0.2),
        ])

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
        import torchvision.transforms.functional as TF
        import time

        buffer: list[Image.Image] = []
        page_label_counter = 0

        # Phase 1: Fill the buffer with an initial batch of images.
        # We use a retry loop so a single timeout doesn't kill the whole run.
        logger.info("🚀 Filling initial buffer for embedder dataset (split=%s)...", self.split)
        if self.streaming:
            dataset = load_dataset(self.dataset_path, split=self.split, streaming=True)
            stream_iter = iter(dataset)
        else:
            dataset = self._local_ds
            stream_iter = iter(dataset)
        fill_target = min(50, self.buffer_size)
        retries = 0
        max_retries = 10
        while len(buffer) < fill_target and retries < max_retries:
            try:
                item = next(stream_iter)
                img = item["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buffer.append(img)
                retries = 0  # reset on success
            except StopIteration:
                break
            except Exception as e:
                retries += 1
                wait = min(2 ** retries, 30)
                logger.warning("⚠️  Buffer fill error (retry %d/%d, waiting %ds): %s", retries, max_retries, wait, e)
                time.sleep(wait)
                # Recreate the stream iterator to recover from broken connections
                try:
                    stream_iter = iter(self.dataset)
                except Exception:
                    pass

        logger.info("✅ Buffer filled with %d images. Starting training sample generation.", len(buffer))

        if len(buffer) < self.pages_per_batch:
            logger.error("❌ Buffer has only %d images, need at least %d (pages_per_batch). Aborting.", len(buffer), self.pages_per_batch)
            return

        # Phase 2: Yield training samples indefinitely from the buffer.
        # Simultaneously try to grow the buffer in the background by pulling
        # one new image from the stream every N yields.
        yields_since_fetch = 0
        fetch_every = 5  # try to fetch a new image every 5 yields

        while True:
            # Sample pages_per_batch different pages from the buffer
            sampled_pages = random.sample(buffer, min(self.pages_per_batch, len(buffer)))

            for page_img in sampled_pages:
                all_strips = self._process_image(page_img)

                if len(all_strips) < self.strips_per_page:
                    continue

                sampled_strips = random.sample(all_strips, self.strips_per_page)

                for strip in sampled_strips:
                    # Apply augmentation to PIL strip
                    augmented_strip = self.strip_augment(strip)
                    tensor = preprocess_single_strip(augmented_strip)
                    yield tensor, page_label_counter

                page_label_counter += 1

            # Periodically try to add fresh images to the buffer
            yields_since_fetch += 1
            if yields_since_fetch >= fetch_every:
                yields_since_fetch = 0
                try:
                    item = next(stream_iter)
                    img = item["image"]
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buffer.append(img)
                    # Cap buffer size
                    if len(buffer) > self.buffer_size:
                        buffer.pop(random.randint(0, len(buffer) // 2))
                except StopIteration:
                    # Stream exhausted, restart it
                    try:
                        stream_iter = iter(dataset)
                    except Exception:
                        pass
                except Exception:
                    # Network error — just keep training from existing buffer
                    pass


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
    parser.add_argument("--pages-per-batch", type=int, default=16, help="Pages per batch")
    parser.add_argument("--strips-per-page", type=int, default=6, help="Strips per page")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--train-steps-per-epoch", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--log-interval", type=int, default=50, help="Log progress every N steps")
    parser.add_argument("--local", action="store_true", help="Use local HuggingFace cache instead of streaming the dataset.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")

    # Dataset and DataLoader
    batch_size = args.pages_per_batch * args.strips_per_page

    # Container-aware worker detection (same logic as seam trainer)
    if not args.local:
        num_workers = 0
    else:
        import math as _math
        effective_cpus = os.cpu_count() or 4
        # Check cgroup limits (Docker/Vast.ai)
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                parts = f.read().strip().split()
                if parts[0] != "max":
                    effective_cpus = min(effective_cpus, max(1, _math.ceil(int(parts[0]) / int(parts[1]))))
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read().strip())
            if quota > 0:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                    period = int(f.read().strip())
                effective_cpus = min(effective_cpus, max(1, _math.ceil(quota / period)))
        except (OSError, ValueError):
            pass
        try:
            effective_cpus = min(effective_cpus, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            pass
        cpu_limit = max(1, effective_cpus - 2)
        
        # Read exact container RAM limit via cgroup
        container_ram_mb = 8192  # Fallback to 8GB
        try:
            with open("/sys/fs/cgroup/memory.max") as f:
                val = f.read().strip()
                if val != "max":
                    container_ram_mb = int(val) // (1024 * 1024)
        except OSError:
            try:
                with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                    val = f.read().strip()
                    if val != "9223372036854771712":
                        container_ram_mb = int(val) // (1024 * 1024)
            except OSError:
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                container_ram_mb = int(line.split()[1]) // 1024
                                break
                except OSError:
                    pass
                    
        # Reserve 4 GB for OS/GPU/main process, rest is for workers
        usable_mb = max(0, container_ram_mb - 4096)
        ram_limit = max(1, usable_mb // 500)
        
        num_workers = min(cpu_limit, ram_limit)
        logger.info("🧠 Auto-scaled workers: effective_cpus=%d, container_ram=%dMB → allowed by CPU:%d, by RAM:%d → using %d", 
                    effective_cpus, container_ram_mb, cpu_limit, ram_limit, num_workers)

    train_dataset = MultiPageStripDataset(
        split="train", 
        pages_per_batch=args.pages_per_batch, 
        strips_per_page=args.strips_per_page,
        streaming=not args.local
    )
    val_dataset = MultiPageStripDataset(
        split="test", 
        pages_per_batch=args.pages_per_batch, 
        strips_per_page=args.strips_per_page,
        streaming=not args.local
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=min(2, num_workers),
        pin_memory=True,
    )

    # Model, Loss, Optimizer
    model = PageEmbeddingNet(embedding_dim=128).to(device)
    criterion = SupConLoss(temperature=0.12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.amp.GradScaler(device.type)

    start_epoch = 0
    best_gap = -float("inf")

    if args.resume:
        logger.info(f"🔄 Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_gap = checkpoint.get("best_gap", best_gap)

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
                embeddings = model(images, use_projection=True)
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
            "best_gap": best_gap,
        }

        # Save latest model
        latest_path = Path(args.output_dir) / "latest_checkpoint.pth"
        torch.save(ckpt_payload, latest_path)
        logger.info(f"💾 Saved latest checkpoint → {latest_path}")

        # Save best model by separation gap
        if sim_gap > best_gap:
            best_gap = sim_gap
            ckpt_payload["best_gap"] = best_gap
            best_path = Path(args.output_dir) / "best_page_embedder.pth"
            torch.save(ckpt_payload, best_path)
            logger.info(f"🏆 New best separation gap ({best_gap:+.4f})! Saved model → {best_path}")


def _cleanup_and_exit(signum, frame):
    """Kill the entire process group (main + all DataLoader workers) on signal."""
    logger.info("🛑 Received signal %d, killing all workers and exiting...", signum)
    os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)


if __name__ == "__main__":
    os.setpgrp()
    signal.signal(signal.SIGINT, _cleanup_and_exit)
    signal.signal(signal.SIGTERM, _cleanup_and_exit)
    main()
