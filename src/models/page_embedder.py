"""Page embedding network for strip-to-page clustering.

This module contains the PageEmbeddingNet used to produce strip embeddings
that cluster by page identity. Trained with Supervised Contrastive Loss.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


class PageEmbeddingNet(nn.Module):
    """Encodes a single strip into a 128-d embedding vector.

    Strips from the same page should cluster together in embedding space,
    while strips from different pages should be far apart.

    Architecture:
        ResNet18 backbone (remove FC) → GlobalAvgPool → FC(512→256) → ReLU → FC(256→128)
        Output is L2-normalized for cosine similarity.

    Trained with Supervised Contrastive Loss (SupCon, Khosla et al. 2020).
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        # Remove the final FC layer, keep everything up to avgpool
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of strip images, shape (B, 3, 224, 224).

        Returns:
            L2-normalized embeddings, shape (B, embedding_dim).
        """
        features = self.encoder(x)  # (B, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512)
        embeddings = self.projector(features)  # (B, embedding_dim)
        return F.normalize(embeddings, dim=1)  # L2-normalize


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    For a batch of (embedding, label) pairs, pulls same-label embeddings
    together and pushes different-label embeddings apart.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            embeddings: L2-normalized embeddings, shape (B, D).
            labels: Page labels for each strip, shape (B,).

        Returns:
            Scalar loss.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Pairwise cosine similarity (embeddings are already L2-normalized)
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # Mask: 1 where labels match, 0 otherwise
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float().to(device)

        # Remove self-similarity from both mask and logits
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask = mask.masked_fill(self_mask, 0)

        # For numerical stability, subtract max from logits
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(self_mask, 0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        positive_count = mask.sum(dim=1)
        # Avoid division by zero for strips with no positives in the batch
        valid = positive_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (positive_count + 1e-12)
        loss = -mean_log_prob[valid].mean()

        return loss


def load_page_embedder(
    model_path: str | Path,
    device: torch.device | None = None,
    embedding_dim: int = 128,
) -> tuple[PageEmbeddingNet, torch.device]:
    """Load a trained PageEmbeddingNet from a checkpoint.

    Args:
        model_path: Path to the .pth checkpoint file.
        device: Target device. Auto-detects CUDA if None.
        embedding_dim: Embedding dimension (must match the saved model).

    Returns:
        Tuple of (model, device). Returns (None, device) if not found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PageEmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error("Page embedder model not found: %s", model_path)
        return None, device

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        
        if "model_state_dict" in checkpoint:
            state_dict_raw = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict_raw = checkpoint["model"]
        else:
            state_dict_raw = checkpoint
            
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict_raw.items()
        }
        model.load_state_dict(state_dict)
        model.to(device).eval()
        logger.info("✅ Loaded PageEmbeddingNet from %s", model_path)
        return model, device
    except Exception as e:
        logger.error("❌ Error loading page embedder from %s: %s", model_path, e)
        return None, device
