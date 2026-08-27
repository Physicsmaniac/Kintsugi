"""Shared preprocessing pipeline for strip pair analysis.

This module consolidates the image preprocessing logic that was previously
duplicated across app.py, evaluate.py, legacy_solver.py, and train.py.
All scoring and inference code should use these functions.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

logger = logging.getLogger(__name__)

# --- Constants ---
TRAINING_STRIP_WIDTH = 48
CROP_SIZE = 224

# --- Shared Transform ---
normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomCrop((CROP_SIZE, CROP_SIZE), pad_if_needed=True, fill=255),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Validation / inference transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.RandomCrop((CROP_SIZE, CROP_SIZE), pad_if_needed=True, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# --- Preprocessing Functions ---

def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """Convert an image to grayscale, then back to 3-channel RGB.

    This makes the model robust against color/lighting artifacts while
    maintaining the 3-channel input expected by ResNet.

    Args:
        img: Input PIL image.

    Returns:
        Grayscale image in RGB mode (3 identical channels).
    """
    return ImageOps.grayscale(img).convert("RGB")


def resize_to_training_width(
    img: Image.Image, width: int = TRAINING_STRIP_WIDTH
) -> Image.Image:
    """Resize a strip to the training width while maintaining aspect ratio.

    Args:
        img: Input PIL image (a single strip).
        width: Target width in pixels (default: 32).

    Returns:
        Resized PIL image.
    """
    w, h = img.size
    if w == 0:
        return img
    scale = width / w
    new_h = max(1, int(h * scale))
    return img.resize((width, new_h), Image.Resampling.BILINEAR)


def pad_to_model_size(
    img_crop: Image.Image, size: int = CROP_SIZE
) -> Image.Image:
    """Center an image crop on a white canvas of the target size.

    Args:
        img_crop: Input image crop.
        size: Target canvas size (square, default: 224).

    Returns:
        Padded image on white background, size × size pixels.
    """
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset_x = (size - img_crop.size[0]) // 2
    offset_y = (size - img_crop.size[1]) // 2
    canvas.paste(img_crop, (offset_x, offset_y))
    return canvas


def create_seam_pair(
    img_a: Image.Image,
    img_b: Image.Image,
    training_strip_width: int = TRAINING_STRIP_WIDTH,
) -> Image.Image:
    """Create a combined seam image from two strips.

    Resizes both strips to the training width, crops to the minimum height,
    and concatenates them horizontally.

    Args:
        img_a: Left strip (PIL image).
        img_b: Right strip (PIL image).
        training_strip_width: Width to resize each strip to.

    Returns:
        Combined image, (training_strip_width * 2) × min_height pixels.
    """
    s1 = resize_to_training_width(img_a, training_strip_width)
    s2 = resize_to_training_width(img_b, training_strip_width)

    min_h = min(s1.size[1], s2.size[1])
    s1 = s1.crop((0, 0, training_strip_width, min_h))
    s2 = s2.crop((0, 0, training_strip_width, min_h))

    combined = Image.new("RGB", (training_strip_width * 2, min_h))
    combined.paste(s1, (0, 0))
    combined.paste(s2, (training_strip_width, 0))
    return combined


def get_crops(
    combined_img: Image.Image, crop_size: int = CROP_SIZE
) -> list[Image.Image]:
    """Apply the 3-crop strategy to a combined seam image.

    If the image is taller than crop_size, takes top, middle, and bottom crops.
    Otherwise, pads the single image and duplicates it 3 times.

    This ensures robustness to varying strip heights — tall strips get evaluated
    at multiple vertical positions.

    Args:
        combined_img: Combined seam image (two strips side-by-side).
        crop_size: Height of each crop (default: 224).

    Returns:
        List of 3 padded crops, each crop_size × crop_size pixels.
    """
    w, h = combined_img.size

    if h <= crop_size:
        # Image fits in one crop — pad and triplicate
        padded = pad_to_model_size(combined_img, crop_size)
        return [padded, padded, padded]
    else:
        # Take top, middle, bottom crops
        c1 = combined_img.crop((0, 0, w, crop_size))
        mid_y = (h - crop_size) // 2
        c2 = combined_img.crop((0, mid_y, w, mid_y + crop_size))
        c3 = combined_img.crop((0, h - crop_size, w, h))
        return [pad_to_model_size(c, crop_size) for c in [c1, c2, c3]]


def preprocess_pair(
    img_a: Image.Image,
    img_b: Image.Image,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Full preprocessing pipeline for a single strip pair.

    Combines create_seam_pair → get_crops → normalize into a single call.

    Args:
        img_a: Left strip.
        img_b: Right strip.
        device: Target device for the output tensor.

    Returns:
        Tensor of shape (3, 3, 224, 224) — 3 crops, each normalized.
    """
    combined = create_seam_pair(img_a, img_b)
    crops = get_crops(combined)
    tensors = [normalize_transform(c) for c in crops]
    batch = torch.stack(tensors)
    if device is not None:
        batch = batch.to(device)
    return batch


def preprocess_single_strip(
    img: Image.Image,
    size: int = CROP_SIZE,
) -> torch.Tensor:
    """Preprocess a single strip for the PageEmbeddingNet.

    Resizes the strip to training width, pads to model size, and normalizes.

    Args:
        img: Input strip image.
        size: Target canvas size.

    Returns:
        Normalized tensor of shape (3, size, size).
    """
    # 1. Resize width to TRAINING_STRIP_WIDTH maintaining aspect ratio
    target_w = TRAINING_STRIP_WIDTH
    new_h = max(1, img.height * target_w // max(1, img.width))
    strip = img.resize((target_w, new_h), Image.Resampling.LANCZOS)
    
    # 2. Pad to 224x224 (if smaller) or add 0 padding if larger
    pad_w = max(0, size - strip.width)
    pad_h = max(0, size - strip.height)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    padded_strip = ImageOps.expand(strip, border=(pad_left, pad_top, pad_right, pad_bottom), fill="white")
    
    # 3. Squish to exactly 224x224 if it's taller than 224
    if padded_strip.size != (size, size):
        padded_strip = padded_strip.resize((size, size), Image.Resampling.LANCZOS)
        
    return normalize_transform(padded_strip)


def create_seam_edge_pair(
    img_a: Image.Image,
    img_b: Image.Image,
    edge_width: int = TRAINING_STRIP_WIDTH,
) -> Image.Image:
    """Create a seam pair using only the edge regions of two strips.

    Instead of resizing full strips, this function takes the rightmost
    ``edge_width`` pixels of strip A and the leftmost ``edge_width``
    pixels of strip B, preserving seam boundary detail at higher resolution.

    This is the recommended approach for training when strips are wider
    than ``TRAINING_STRIP_WIDTH``, as it avoids the information loss from
    aggressive downscaling.

    Args:
        img_a: Left strip (PIL image).
        img_b: Right strip (PIL image).
        edge_width: Number of pixels to take from each strip's seam edge.

    Returns:
        Combined image, (edge_width * 2) × min_height pixels.
    """
    w_a, h_a = img_a.size
    w_b, h_b = img_b.size

    # Take the right edge of strip A
    if w_a > edge_width:
        left_edge = img_a.crop((w_a - edge_width, 0, w_a, h_a))
    else:
        left_edge = resize_to_training_width(img_a, edge_width)

    # Take the left edge of strip B
    if w_b > edge_width:
        right_edge = img_b.crop((0, 0, edge_width, h_b))
    else:
        right_edge = resize_to_training_width(img_b, edge_width)

    min_h = min(left_edge.size[1], right_edge.size[1])
    left_edge = left_edge.crop((0, 0, edge_width, min_h))
    right_edge = right_edge.crop((0, 0, edge_width, min_h))

    combined = Image.new("RGB", (edge_width * 2, min_h))
    combined.paste(left_edge, (0, 0))
    combined.paste(right_edge, (edge_width, 0))
    return combined

