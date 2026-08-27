"""Seam compatibility model for strip pair classification.

This module contains the SeamResNet architecture used to predict whether
two vertical strips are adjacent (i.e., strip B goes to the right of strip A).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class SeamResNet(nn.Module):
    """ResNet18-based binary classifier for seam compatibility.

    Takes a 224×224 image containing two strips side-by-side (64px wide total,
    centered on a white canvas) and outputs a single logit indicating whether
    the right strip is the true right neighbor of the left strip.

    Architecture:
        ResNet18 backbone → FC(512→256) → BN → ReLU → Dropout(0.3)
                           → FC(256→128) → BN → ReLU → Dropout(0.2)
                           → FC(128→1)

    The deeper head provides more representational capacity for
    ranking-aware scoring needed by the ATSP solver.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.cnn = models.resnet18(weights=weights)
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of images, shape (B, 3, 224, 224).

        Returns:
            Raw logits, shape (B, 1).
        """
        return self.cnn(x)


def load_seam_model(
    model_path: str | Path,
    device: torch.device | None = None,
    pretrained_backbone: bool = False,
) -> tuple[SeamResNet, torch.device]:
    """Load a trained SeamResNet from a checkpoint file.

    Handles DataParallel state dict keys (strips 'module.' prefix).

    Args:
        model_path: Path to the .pth checkpoint file.
        device: Target device. Auto-detects CUDA if None.
        pretrained_backbone: Whether to initialize with ImageNet weights
            before loading checkpoint (usually False for inference).

    Returns:
        Tuple of (model, device). Model is in eval mode on the target device.
        Returns (None, device) if the model file doesn't exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SeamResNet(pretrained=pretrained_backbone)
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return None, device

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        
        if "model_state_dict" in checkpoint:
            state_dict_raw = checkpoint["model_state_dict"]
        else:
            state_dict_raw = checkpoint
            
        # Handle DataParallel state dicts
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict_raw.items()
        }
        model.load_state_dict(state_dict)
        model.to(device).eval()
        logger.info("✅ Loaded SeamResNet from %s", model_path)
        return model, device
    except Exception as e:
        logger.error("❌ Error loading model from %s: %s", model_path, e)
        return None, device
