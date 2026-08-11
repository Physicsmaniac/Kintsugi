"""Pairwise strip scoring for shredded document reconstruction.

Computes an n×n asymmetric score matrix where score[i,j] represents the
probability that strip j belongs immediately to the RIGHT of strip i.

Scoring pipeline per pair (i, j):
    1. Convert PIL images to grayscale → 3-channel RGB
    2. Resize each strip to TRAINING_STRIP_WIDTH (32px) wide, preserving aspect ratio
    3. Horizontally concatenate resized strip_i and strip_j → 64px wide seam image
    4. 3-crop strategy: top / middle / bottom 224px crops (or padded duplicates)
    5. Centre-pad each crop on a 224×224 white canvas
    6. Normalise with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
    7. Score = average sigmoid over the 3 crops
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    create_seam_pair,
    get_crops,
    normalize_transform,
    pad_to_model_size,
    resize_to_training_width,
    to_grayscale_rgb,
)

logger = logging.getLogger(__name__)

TRAINING_STRIP_WIDTH: int = 32
CROP_SIZE: int = 224
NUM_CROPS: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_strip(img: Image.Image, width: int = TRAINING_STRIP_WIDTH) -> Image.Image:
    """Convert a PIL image to grayscale-RGB and resize to training width."""
    grey_rgb = to_grayscale_rgb(img)
    return resize_to_training_width(grey_rgb, width=width)


def _crops_to_tensor(crops: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL crops into a single (N, 3, 224, 224) tensor."""
    tensors: list[torch.Tensor] = []
    for crop in crops:
        padded = pad_to_model_size(crop, size=CROP_SIZE)
        t = normalize_transform(padded)  # (3, H, W)
        tensors.append(t)
    return torch.stack(tensors, dim=0)


def _score_pair(
    model: nn.Module,
    device: torch.device,
    img_a: Image.Image,
    img_b: Image.Image,
) -> tuple[float, float]:
    """Return (score, logit) for one ordered pair.

    *score* is the averaged sigmoid probability; *logit* is the averaged
    raw model output (pre-sigmoid), useful for temperature scaling later.
    """
    combined = create_seam_pair(img_a, img_b, training_strip_width=TRAINING_STRIP_WIDTH)
    crops = get_crops(combined, crop_size=CROP_SIZE)

    # Ensure exactly NUM_CROPS crops (duplicate if only 1)
    while len(crops) < NUM_CROPS:
        crops.append(crops[0])

    batch = _crops_to_tensor(crops).to(device)  # (3, 3, 224, 224)

    with torch.no_grad():
        logits = model(batch).squeeze(-1)  # (3,)
        probs = torch.sigmoid(logits)

    mean_logit = logits.mean().item()
    mean_prob = probs.mean().item()
    return mean_prob, mean_logit


# ---------------------------------------------------------------------------
# Simple loop-based scoring
# ---------------------------------------------------------------------------

def compute_score_matrix(
    model: nn.Module,
    device: torch.device,
    images: list[Image.Image],
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full n×n pairwise score matrix.

    Parameters
    ----------
    model : nn.Module
        Binary classifier that outputs a single logit per sample.
    device : torch.device
        Device the model lives on.
    images : list[Image.Image]
        The strip images in their original order/index.
    progress_callback : callable, optional
        Called with a float in [0, 1] to report progress.

    Returns
    -------
    score_matrix : np.ndarray
        n×n matrix of sigmoid probabilities.
    logit_matrix : np.ndarray
        n×n matrix of raw (pre-sigmoid) logits for downstream
        temperature scaling.
    """
    model.eval()
    n = len(images)
    logger.info("Computing %d×%d score matrix (%d pairs)…", n, n, n * (n - 1))

    # Pre-process strips once
    prepared: list[Image.Image] = [_prepare_strip(img) for img in images]

    score_matrix = np.zeros((n, n), dtype=np.float64)
    logit_matrix = np.zeros((n, n), dtype=np.float64)

    total_pairs = n * (n - 1)
    done = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            prob, logit = _score_pair(model, device, prepared[i], prepared[j])
            score_matrix[i, j] = prob
            logit_matrix[i, j] = logit
            done += 1
            if progress_callback is not None and done % max(1, total_pairs // 100) == 0:
                progress_callback(done / total_pairs)

    if progress_callback is not None:
        progress_callback(1.0)

    logger.info("Score matrix complete. Range [%.4f, %.4f]",
                score_matrix.min(), score_matrix.max())
    return score_matrix, logit_matrix


# ---------------------------------------------------------------------------
# DataLoader-batched scoring
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """Dataset that yields pre-processed crop tensors for every ordered pair.

    Each item is a dict with keys:
        ``i``, ``j``   – pair indices
        ``crops``      – tensor of shape (NUM_CROPS, 3, 224, 224)
    """

    def __init__(self, images: list[Image.Image]) -> None:
        super().__init__()
        self.prepared: list[Image.Image] = [_prepare_strip(img) for img in images]
        self.n = len(images)
        # Build an index of (i, j) pairs excluding diagonal
        self.pairs: list[tuple[int, int]] = [
            (i, j) for i in range(self.n) for j in range(self.n) if i != j
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        i, j = self.pairs[idx]
        combined = create_seam_pair(
            self.prepared[i], self.prepared[j],
            training_strip_width=TRAINING_STRIP_WIDTH,
        )
        crops = get_crops(combined, crop_size=CROP_SIZE)
        while len(crops) < NUM_CROPS:
            crops.append(crops[0])
        crop_tensor = _crops_to_tensor(crops)  # (NUM_CROPS, 3, 224, 224)
        return {"i": i, "j": j, "crops": crop_tensor}


def _collate_fn(
    batch: list[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    """Custom collate: stack crops across pairs."""
    return {
        "i": torch.tensor([b["i"] for b in batch], dtype=torch.long),
        "j": torch.tensor([b["j"] for b in batch], dtype=torch.long),
        "crops": torch.stack([b["crops"] for b in batch], dim=0),
        # crops shape: (B, NUM_CROPS, 3, 224, 224)
    }


def compute_score_matrix_batched(
    model: nn.Module,
    device: torch.device,
    images: list[Image.Image],
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """DataLoader-batched version of :func:`compute_score_matrix`.

    Substantially faster on GPU because crops for multiple pairs are
    forwarded through the model together.

    Parameters
    ----------
    model : nn.Module
        Binary classifier outputting a single logit.
    device : torch.device
        Target compute device.
    images : list[Image.Image]
        Strip images.
    batch_size : int
        Number of *pairs* per batch (each pair has NUM_CROPS crops).
    num_workers : int
        DataLoader workers.

    Returns
    -------
    score_matrix : np.ndarray
        n×n sigmoid probability matrix.
    logit_matrix : np.ndarray
        n×n raw logit matrix.
    """
    model.eval()
    n = len(images)
    logger.info(
        "Computing %d×%d score matrix (batched, bs=%d, workers=%d)…",
        n, n, batch_size, num_workers,
    )

    dataset = InferenceDataset(images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    score_matrix = np.zeros((n, n), dtype=np.float64)
    logit_matrix = np.zeros((n, n), dtype=np.float64)

    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            idx_i = batch["i"]  # (B,)
            idx_j = batch["j"]  # (B,)
            crops = batch["crops"].to(device)  # (B, NUM_CROPS, 3, 224, 224)

            b, nc, c, h, w = crops.shape
            flat_crops = crops.view(b * nc, c, h, w)  # (B*NUM_CROPS, 3, 224, 224)

            logits = model(flat_crops).squeeze(-1)  # (B*NUM_CROPS,)
            logits = logits.view(b, nc)  # (B, NUM_CROPS)
            probs = torch.sigmoid(logits)  # (B, NUM_CROPS)

            mean_logits = logits.mean(dim=1).cpu().numpy()  # (B,)
            mean_probs = probs.mean(dim=1).cpu().numpy()  # (B,)

            for k in range(b):
                ii = idx_i[k].item()
                jj = idx_j[k].item()
                score_matrix[ii, jj] = mean_probs[k]
                logit_matrix[ii, jj] = mean_logits[k]

            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                logger.info(
                    "  batch %d/%d (%.0f%%)",
                    batch_idx + 1, total_batches,
                    100.0 * (batch_idx + 1) / total_batches,
                )

    logger.info(
        "Batched score matrix complete. Range [%.4f, %.4f]",
        score_matrix.min(), score_matrix.max(),
    )
    return score_matrix, logit_matrix
