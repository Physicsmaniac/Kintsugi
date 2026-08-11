"""Document shredder for generating test sets.

Shreds PDF and image documents into vertical strips with ground truth metadata
for evaluation. Supports both single-page and multi-page shredding.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def shred_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    num_strips: int = 10,
    max_pages: int | None = None,
    dpi: int = 200,
    shuffle: bool = True,
    seed: int | None = None,
) -> dict:
    """Shred a multi-page PDF into randomized vertical strips.

    Each page is divided into equal-width vertical strips. All strips across
    all pages are shuffled together and saved with ground truth metadata.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to save strips and ground truth.
        num_strips: Number of strips per page.
        max_pages: Maximum pages to process (None = all pages).
        dpi: Rendering DPI for PDF pages.
        shuffle: Whether to shuffle strips across pages.
        seed: Random seed for reproducibility.

    Returns:
        Ground truth dict mapping filename → {page, index}.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF shredding. "
            "Install it with: pip install pymupdf"
        )

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)

    doc = fitz.open(str(pdf_path))
    pages_to_process = len(doc) if max_pages is None else min(len(doc), max_pages)
    logger.info("🔥 Shredding %d pages from %s (%d strips/page)...",
                pages_to_process, pdf_path.name, num_strips)

    strips_metadata = []

    for page_num in range(pages_to_process):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)

        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_data.reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w = img.shape[:2]
        strip_width = w // num_strips

        for i in range(num_strips):
            x_start = i * strip_width
            x_end = w if (i == num_strips - 1) else (i + 1) * strip_width
            strip = img[:, x_start:x_end]

            strips_metadata.append({
                "image": strip,
                "original_page": page_num,
                "original_index": i,
            })

        logger.info("   📄 Processed page %d/%d", page_num + 1, pages_to_process)

    doc.close()

    # Shuffle all strips across pages
    if shuffle:
        logger.info("🌪️  Mixing %d strips...", len(strips_metadata))
        random.shuffle(strips_metadata)

    # Save strips and ground truth
    ground_truth = {}
    for idx, item in enumerate(strips_metadata):
        fname = f"strip_{idx:04d}.jpg"
        filepath = output_dir / fname
        cv2.imwrite(str(filepath), item["image"])

        ground_truth[fname] = {
            "page": item["original_page"],
            "index": item["original_index"],
        }

    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    logger.info("✅ Saved %d strips to %s", len(strips_metadata), output_dir)
    return ground_truth


def shred_image(
    image_path: str | Path,
    output_dir: str | Path,
    num_strips: int = 10,
    shuffle: bool = True,
    seed: int | None = None,
) -> dict:
    """Shred a single image into randomized vertical strips.

    Args:
        image_path: Path to the input image file.
        output_dir: Directory to save strips and ground truth.
        num_strips: Number of strips to create.
        shuffle: Whether to shuffle the strip order.
        seed: Random seed for reproducibility.

    Returns:
        Ground truth dict mapping filename → {page: 0, index}.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    strip_width = w // num_strips

    strips = []
    for i in range(num_strips):
        x_start = i * strip_width
        x_end = w if (i == num_strips - 1) else (i + 1) * strip_width
        strip = img[:, x_start:x_end]
        strips.append({"image": strip, "index": i})

    if shuffle:
        random.shuffle(strips)

    ground_truth = {}
    for idx, item in enumerate(strips):
        fname = f"strip_{idx:04d}.jpg"
        filepath = output_dir / fname
        cv2.imwrite(str(filepath), item["image"])
        ground_truth[fname] = {"page": 0, "index": item["index"]}

    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    logger.info("✅ Shredded image into %d strips → %s", num_strips, output_dir)
    return ground_truth
