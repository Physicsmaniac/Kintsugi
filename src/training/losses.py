"""Ranking-aware losses for seam compatibility training.

Provides InfoNCE and combined BCE+Ranking losses that train the model to
produce scores suitable for the ATSP solver. Unlike plain BCE which only
learns a decision boundary, these losses optimise the *relative ordering*
of scores across candidates.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCESeamLoss(nn.Module):
    """InfoNCE-style loss for seam compatibility ranking.

    Given a batch of (pair_tensor, label) samples, this loss treats each
    positive pair as the "correct" match and all negative pairs in the
    batch as distractors. The model must rank the positive higher than
    all negatives.

    This is much better calibrated for the ATSP solver than BCE, because
    it directly optimises the relative ranking of scores rather than
    just a binary threshold.

    Parameters
    ----------
    temperature : float
        Softmax temperature. Lower = sharper ranking. Default 0.1.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the InfoNCE loss.

        For each positive sample in the batch, we compute a softmax over
        the positive logit vs all negative logits, and maximise the
        log-probability of the positive.

        Args:
            logits: Raw model outputs, shape (B,).
            labels: Binary labels, shape (B,). 1.0 = positive, 0.0 = negative.

        Returns:
            Scalar loss.
        """
        device = logits.device
        positive_mask = labels > 0.5
        negative_mask = labels < 0.5

        n_pos = positive_mask.sum().item()
        n_neg = negative_mask.sum().item()

        if n_pos == 0 or n_neg == 0:
            # Fallback to BCE if batch has no positives or no negatives
            return F.binary_cross_entropy_with_logits(logits, labels)

        positive_logits = logits[positive_mask]  # (n_pos,)
        negative_logits = logits[negative_mask]  # (n_neg,)

        # For each positive, compute softmax against all negatives
        # Shape: (n_pos, 1) vs (1, n_neg) → (n_pos, n_neg+1)
        pos_expanded = positive_logits.unsqueeze(1)  # (n_pos, 1)
        neg_expanded = negative_logits.unsqueeze(0).expand(n_pos, -1)  # (n_pos, n_neg)

        # Concatenate: [positive_logit, neg_1, neg_2, ..., neg_K]
        all_logits = torch.cat([pos_expanded, neg_expanded], dim=1)  # (n_pos, 1+n_neg)
        all_logits = all_logits / self.temperature

        # Target: index 0 is always the positive
        targets = torch.zeros(n_pos, dtype=torch.long, device=device)

        loss = F.cross_entropy(all_logits, targets)
        return loss


class CombinedSeamLoss(nn.Module):
    """Combined BCE + Ranking loss for seam compatibility.

    Mixes the standard binary cross-entropy (for calibrated probabilities)
    with a margin-based ranking loss (for correct relative ordering).

    Parameters
    ----------
    bce_weight : float
        Weight for the BCE component. Default 0.3.
    ranking_weight : float
        Weight for the InfoNCE ranking component. Default 0.7.
    temperature : float
        Temperature for the InfoNCE component.
    """

    def __init__(
        self,
        bce_weight: float = 0.3,
        ranking_weight: float = 0.7,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.ranking_weight = ranking_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ranking_loss = InfoNCESeamLoss(temperature=temperature)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            logits: Raw model outputs, shape (B,).
            labels: Binary labels, shape (B,).

        Returns:
            Scalar combined loss.
        """
        bce = self.bce_loss(logits, labels)
        ranking = self.ranking_loss(logits, labels)
        return self.bce_weight * bce + self.ranking_weight * ranking
