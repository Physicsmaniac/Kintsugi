from __future__ import annotations

"""Training script for SeamResNet strip-pair classifier.

Features
--------
* Mixed-precision training with ``torch.amp`` (non-deprecated API).
* CSV logging of accuracy, loss, precision, recall, and F1.
* Cosine annealing learning-rate scheduler.
* Best-model saving (by validation accuracy) and latest-checkpoint saving.
* Multi-GPU via ``DataParallel``.
* CLI argument parsing with ``argparse``.
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.seam_model import SeamResNet
from src.training.dataset import StreamingShredDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing all training hyper-parameters.
    """
    parser = argparse.ArgumentParser(
        description="Train SeamResNet for strip-pair classification."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Mini-batch size (default: 256)."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)."
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=3000,
        help="Training steps per epoch (default: 3000).",
    )
    parser.add_argument(
        "--val-steps",
        type=int,
        default=400,
        help="Validation steps per epoch (default: 400).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints and logs (default: checkpoints).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker count (default: 8).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------


def _train_transforms() -> transforms.Compose:
    """Build training augmentation pipeline."""
    return transforms.Compose(
        [
            transforms.RandomCrop(224),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _val_transforms() -> transforms.Compose:
    """Build validation transform pipeline (deterministic)."""
    return transforms.Compose(
        [
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Execute the full training procedure.

    Parameters
    ----------
    args:
        Parsed CLI arguments (see :func:`parse_args`).
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("🖥️  Using device: %s", device)

    # ---- model --------------------------------------------------------
    model = SeamResNet()
    start_epoch = 0

    if args.resume is not None and os.path.isfile(args.resume):
        logger.info("🔄 Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Strip 'module.' prefix if saved from DataParallel
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("module.")] = v
        model.load_state_dict(cleaned)
        start_epoch = ckpt.get("epoch", 0)
        logger.info("✅ Loaded checkpoint (epoch %d)", start_epoch)

    model.to(device)

    if torch.cuda.device_count() > 1:
        logger.info("🔀 Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    # ---- data ---------------------------------------------------------
    train_ds = StreamingShredDataset(split="train", transform=_train_transforms())
    val_ds = StreamingShredDataset(split="test", transform=_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- optimiser & scheduler ----------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Restore optimizer state if resuming
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("✅ Restored optimizer state.")

    best_val_acc: float = 0.0

    # ---- CSV log ------------------------------------------------------
    csv_path = output_dir / "training_log.csv"
    csv_fields = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "precision",
        "recall",
        "f1",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_fields)
    logger.info("📝 CSV log: %s", csv_path)

    # ---- epoch loop ---------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        logger.info("🏋️  Epoch %d / %d", epoch + 1, args.epochs)

        # ---- train phase ----------------------------------------------
        model.train()
        train_loss_sum: float = 0.0
        train_correct: int = 0
        train_total: int = 0
        train_iter = iter(train_loader)

        for step in range(1, args.steps_per_epoch + 1):
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(images).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            preds = (logits.detach() > 0.0).float()
            train_correct += (preds == labels).sum().item()
            train_total += batch_size

            if step % 500 == 0:
                running_acc = train_correct / max(1, train_total)
                running_loss = train_loss_sum / max(1, train_total)
                logger.info(
                    "   📈 Step %d/%d – loss: %.4f  acc: %.4f",
                    step,
                    args.steps_per_epoch,
                    running_loss,
                    running_acc,
                )

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ---- validation phase -----------------------------------------
        model.eval()
        val_loss_sum: float = 0.0
        val_correct: int = 0
        val_total: int = 0
        all_preds: list[int] = []
        all_labels: list[int] = []
        val_iter = iter(val_loader)

        with torch.no_grad():
            for step in range(1, args.val_steps + 1):
                try:
                    images, labels = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    images, labels = next(val_iter)

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                preds = (logits > 0.0).float()
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

                all_preds.extend(preds.cpu().int().tolist())
                all_labels.extend(labels.cpu().int().tolist())

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        # sklearn metrics
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        logger.info(
            "📊 Epoch %d results – train_loss: %.4f | train_acc: %.4f | "
            "val_loss: %.4f | val_acc: %.4f | P: %.4f | R: %.4f | F1: %.4f",
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            precision,
            recall,
            f1,
        )

        # ---- CSV append -----------------------------------------------
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{precision:.6f}",
                    f"{recall:.6f}",
                    f"{f1:.6f}",
                ]
            )

        # ---- scheduler step -------------------------------------------
        scheduler.step()

        # ---- checkpointing --------------------------------------------
        ckpt_payload = {
            "epoch": epoch + 1,
            "model_state_dict": (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }

        # Always save latest
        latest_path = output_dir / "latest_checkpoint.pt"
        torch.save(ckpt_payload, latest_path)
        logger.info("💾 Saved latest checkpoint → %s", latest_path)

        # Save best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = output_dir / "best_model.pt"
            torch.save(ckpt_payload, best_path)
            logger.info("🏆 New best model (val_acc=%.4f) → %s", val_acc, best_path)

    logger.info("✅ Training complete. Best val_acc: %.4f", best_val_acc)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
