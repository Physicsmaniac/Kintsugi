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
import gc
import logging
import os
import signal
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

# Suppress noisy HTTP probe logs from HuggingFace/httpx internals
for _noisy in ("httpx", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

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
        "--batch-size", type=int, default=1024, help="Mini-batch size (default: 1024)."
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
        default=0,
        help="DataLoader worker count (0 = auto-detect from CPU cores).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Step interval for progress logging (default: 50).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local HuggingFace cache instead of streaming the dataset.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Container-aware CPU detection
# -----------------------------------------------------------------------


def _get_container_cpus() -> int:
    """Return the number of CPUs actually available to this process.

    In Docker/Vast.ai containers, ``os.cpu_count()`` reports the *host*
    machine's full thread count (e.g. 128 on an EPYC), not the cores
    allocated to the container.  This causes DataLoader to spawn far too
    many workers, exhausting the container's RAM/CPU quota.

    We check, in order:
    1. cgroup v2  ``cpu.max``  (quota / period)
    2. cgroup v1  ``cpu.cfs_quota_us / cpu.cfs_period_us``
    3. ``os.sched_getaffinity(0)``  (respects ``--cpuset-cpus``)
    4. ``os.cpu_count()``  (bare-metal fallback)

    The returned value is the *minimum* of all readable signals, which
    gives the tightest real constraint.
    """
    import math
    candidates: list[int] = []

    # --- cgroup v2 -------------------------------------------------------
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if parts[0] != "max":            # "max" means unlimited
                quota, period = int(parts[0]), int(parts[1])
                candidates.append(max(1, math.ceil(quota / period)))
    except (OSError, ValueError, IndexError):
        pass

    # --- cgroup v1 -------------------------------------------------------
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        if quota > 0:                        # -1 means unlimited
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read().strip())
            candidates.append(max(1, math.ceil(quota / period)))
    except (OSError, ValueError):
        pass

    # --- CPU affinity (respects --cpuset-cpus) ----------------------------
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass  # Not available on all platforms

    # --- bare-metal fallback ----------------------------------------------
    fallback = os.cpu_count() or 4
    if not candidates:
        return fallback

    result = min(candidates)
    if result != fallback:
        logger.info(
            "🐳 Container CPU detection: host reports %d cores, "
            "container limited to %d",
            fallback, result,
        )
    return result


# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------


def _train_transforms():
    """Transforms for training: random crop, white pad, and grayscale jitter."""
    return transforms.Compose(
        [
            transforms.RandomCrop((224, 64), pad_if_needed=True, fill=255),
            transforms.Pad((80, 0, 80, 0), fill=255),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _val_transforms():
    """Transforms for validation: center crop and white pad."""
    return transforms.Compose(
        [
            transforms.CenterCrop((224, 64)),
            transforms.Pad((80, 0, 80, 0), fill=255),
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

    # buffer_size per worker: enough for cross-document diversity, low enough
    # to avoid OOM when many workers are running.
    # HuggingFace streaming datasets cannot be forked safely in multiprocess DataLoader
    # due to internal file locks, so we force num_workers=0 when streaming.
    if not args.local:
        actual_num_workers = 0
        logger.info("🔧 Streaming mode: forcing num_workers=0")
    elif args.num_workers > 0:
        actual_num_workers = args.num_workers
    else:
        # Smart auto-detect: scale workers based on container CPUs AND available RAM.
        effective_cpus = _get_container_cpus()
        cpu_limit = max(1, effective_cpus - 2)

        # Estimate available RAM by reading /proc/meminfo (Linux)
        ram_limit = cpu_limit  # fallback: no RAM constraint
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_mb = int(line.split()[1]) // 1024  # kB -> MB
                        # Reserve 4 GB for OS/GPU/model/page-cache, rest for workers.
                        # Each worker costs ~500 MB (buffer + PIL decode + prefetch).
                        usable_mb = max(0, avail_mb - 4096)
                        ram_limit = max(1, usable_mb // 500)
                        break
        except OSError:
            pass  # Not Linux or no /proc — just use cpu_limit

        actual_num_workers = min(cpu_limit, ram_limit)
        logger.info("🧠 Auto-scaled workers: effective_cpus=%d, cpu_limit=%d, ram_limit=%d → using %d",
                    effective_cpus, cpu_limit, ram_limit, actual_num_workers)

    # Validation uses fewer workers to avoid OOM when both loaders overlap
    val_num_workers = min(4, actual_num_workers) if actual_num_workers > 0 else 0

    # Scale buffer per worker so total RAM stays bounded (~2 GB total for buffers)
    buf_per_worker = max(20, 200 // max(1, actual_num_workers))
    logger.info("👷 DataLoader workers: train=%d, val=%d (effective CPUs: %d, buffer/worker: %d)",
                actual_num_workers, val_num_workers, _get_container_cpus(), buf_per_worker)

    train_ds = StreamingShredDataset(
        split="train", transform=_train_transforms(), streaming=not args.local,
        buffer_size=buf_per_worker,
    )
    val_ds = StreamingShredDataset(
        split="test", transform=_val_transforms(), streaming=not args.local,
        buffer_size=buf_per_worker,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=actual_num_workers,
        pin_memory=True,
        prefetch_factor=4 if actual_num_workers > 0 else None,
    )
    # Val DataLoader created lazily per epoch to avoid 2x worker memory overhead
    val_loader = None

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
    file_exists = csv_path.exists()
    with open(csv_path, mode="a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_fields)
    logger.info("📝 CSV log: %s", csv_path)

    # ---- epoch loop ---------------------------------------------------
    import time

    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "════════════════════════════════════════════════════════════════"
        )
        logger.info(
            "🏋️  Epoch %d / %d  [lr: %.2e]", epoch + 1, args.epochs, current_lr
        )
        logger.info(
            "════════════════════════════════════════════════════════════════"
        )

        # ---- train phase ----------------------------------------------
        model.train()
        train_loss_sum: float = 0.0
        train_correct: int = 0
        train_total: int = 0
        train_iter = iter(train_loader)
        epoch_start_time = time.time()
        step_start_time = time.time()
        last_log_step = 0

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

            if step % args.log_interval == 0 or step == args.steps_per_epoch:
                elapsed = time.time() - step_start_time
                steps_passed = step - last_log_step
                samples_per_sec = (steps_passed * batch_size) / max(1e-5, elapsed)
                running_acc = train_correct / max(1, train_total)
                running_loss = train_loss_sum / max(1, train_total)
                batch_loss = loss.item()
                
                logger.info(
                    "   📈 [Train Step %4d/%4d]  batch_loss: %.4f | run_loss: %.4f | run_acc: %.4f (%.2f%%) | %.1f samples/sec",
                    step,
                    args.steps_per_epoch,
                    batch_loss,
                    running_loss,
                    running_acc,
                    running_acc * 100.0,
                    samples_per_sec,
                )
                step_start_time = time.time()
                last_log_step = step

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        train_time = time.time() - epoch_start_time

        # ---- validation phase -----------------------------------------
        logger.info("🔍 Running validation...")

        # Create val DataLoader lazily to avoid OOM from overlapping workers
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=val_num_workers,
            pin_memory=True,
            prefetch_factor=4 if val_num_workers > 0 else None,
        )

        model.eval()
        val_loss_sum: float = 0.0
        val_correct: int = 0
        val_total: int = 0
        all_preds: list[int] = []
        all_labels: list[int] = []
        val_iter = iter(val_loader)
        val_start_time = time.time()

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

                if step % args.log_interval == 0 or step == args.val_steps:
                    v_acc = val_correct / max(1, val_total)
                    v_loss = val_loss_sum / max(1, val_total)
                    logger.info(
                        "   🔍 [Val Step   %4d/%4d]  val_loss: %.4f | val_acc: %.4f (%.2f%%)",
                        step,
                        args.val_steps,
                        v_loss,
                        v_acc,
                        v_acc * 100.0,
                    )

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        val_time = time.time() - val_start_time

        # Free val workers to reclaim memory before next training epoch
        del val_loader, val_iter
        import gc; gc.collect()

        # sklearn metrics
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        logger.info(
            "📊 Epoch %d Summary (Train: %.1fs, Val: %.1fs):",
            epoch + 1,
            train_time,
            val_time,
        )
        logger.info(
            "   ├── Train Loss: %.4f | Train Acc: %.2f%%",
            train_loss,
            train_acc * 100.0,
        )
        logger.info(
            "   ├── Val Loss:   %.4f | Val Acc:   %.2f%%",
            val_loss,
            val_acc * 100.0,
        )
        logger.info(
            "   └── Precision:  %.4f | Recall: %.4f | F1-Score: %.4f",
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
        latest_path = output_dir / "latest_checkpoint.pth"
        torch.save(ckpt_payload, latest_path)
        logger.info("💾 Saved latest checkpoint → %s", latest_path)

        # Save best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = output_dir / "best_seam_model.pth"
            torch.save(ckpt_payload, best_path)
            logger.info(
                "🏆 New best validation accuracy (%.2f%%)! Saved model → %s",
                val_acc * 100.0,
                best_path,
            )

    logger.info("✅ Training complete. Best val_acc: %.4f", best_val_acc)


def _cleanup_and_exit(signum, frame):
    """Kill the entire process group (main + all DataLoader workers) on signal."""
    logger.info("🛑 Received signal %d, killing all workers and exiting...", signum)
    # Kill entire process group so forked DataLoader workers die too
    os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)


if __name__ == "__main__":
    # Create a new process group so we can kill all children at once
    os.setpgrp()
    signal.signal(signal.SIGINT, _cleanup_and_exit)
    signal.signal(signal.SIGTERM, _cleanup_and_exit)

    args = parse_args()
    train(args)
