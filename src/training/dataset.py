from __future__ import annotations

"""Streaming shredded-document pair dataset for seam classification.

Streams document images from HuggingFace ``chainyo/rvl-cdip``, maintains a
rolling buffer, and yields horizontally-concatenated strip pairs with labels:

* **Positive (1)** – two adjacent vertical strips.
* **Hard negative (0)** – adjacent strips with a vertical pixel shift.
* **Easy negative (0)** – two non-adjacent strips from the *same* image.
* **Cross-document negative (0)** – strips from *different* images in the buffer.

Ratios are fully configurable via the constructor.
"""

import logging
import random
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

logger = logging.getLogger(__name__)

# Suppress noisy HTTP probe logs from HuggingFace internals
for _noisy_logger in ("httpx", "datasets.load", "huggingface_hub"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_to_max_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """Resize *img* so its width is at most *max_width*, preserving aspect ratio.

    Parameters
    ----------
    img:
        HxWxC or HxW numpy array.
    max_width:
        Maximum allowed width in pixels.

    Returns
    -------
    np.ndarray
        Resized image (or original if already narrow enough).
    """
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_h = max(1, int(h * scale))
    pil = Image.fromarray(img).resize((max_width, new_h), Image.BILINEAR)
    return np.array(pil)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert grayscale or RGBA images to RGB.

    Parameters
    ----------
    img:
        Numpy array of shape HxW, HxWx1, HxWx3, or HxWx4.

    Returns
    -------
    np.ndarray
        HxWx3 RGB image.
    """
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class StreamingShredDataset(IterableDataset):
    """Iterable dataset that streams document images and yields strip pairs.

    Parameters
    ----------
    split:
        HuggingFace dataset split (``"train"`` or ``"test"``).
    transform:
        Torchvision transform pipeline applied to the concatenated strip pair.
    buffer_size:
        Maximum number of images kept in the rolling buffer.
    positive_ratio:
        Fraction of yielded pairs that are positives (adjacent strips).
    hard_neg_ratio:
        Fraction of yielded pairs that are hard negatives (shifted strips).
    easy_neg_ratio:
        Fraction of yielded pairs that are easy negatives (non-adjacent, same image).
    cross_doc_ratio:
        Fraction of yielded pairs that are cross-document negatives.
    max_image_width:
        Images wider than this are down-scaled before strip extraction.
    dataset_path:
        HuggingFace dataset identifier.
    """

    _PAIR_POSITIVE = "positive"
    _PAIR_HARD_NEG = "hard_negative"
    _PAIR_EASY_NEG = "easy_negative"
    _PAIR_CROSS_DOC = "cross_document"

    def __init__(
        self,
        split: str = "train",
        transform: transforms.Compose | None = None,
        buffer_size: int = 1000,
        positive_ratio: float = 0.40,
        hard_neg_ratio: float = 0.20,
        easy_neg_ratio: float = 0.15,
        cross_doc_ratio: float = 0.25,
        max_image_width: int = 400,
        dataset_path: str = "chainyo/rvl-cdip",
        streaming: bool = True,
    ) -> None:
        super().__init__()
        self.split = split
        self.transform = transform
        self.buffer_size = max(1, buffer_size)
        self.max_image_width = max(32, max_image_width)
        self.dataset_path = dataset_path
        self.streaming = streaming

        # Normalise ratios so they sum to 1.0 --------------------------
        total = positive_ratio + hard_neg_ratio + easy_neg_ratio + cross_doc_ratio
        if total <= 0:
            logger.warning(
                "⚠️  All pair ratios are zero – falling back to uniform distribution."
            )
            total = 4.0
            positive_ratio = hard_neg_ratio = easy_neg_ratio = cross_doc_ratio = 1.0
        self.positive_ratio = positive_ratio / total
        self.hard_neg_ratio = hard_neg_ratio / total
        self.easy_neg_ratio = easy_neg_ratio / total
        self.cross_doc_ratio = cross_doc_ratio / total

        # Cumulative thresholds for random selection --------------------
        self._thresh_pos = self.positive_ratio
        self._thresh_hard = self._thresh_pos + self.hard_neg_ratio
        self._thresh_easy = self._thresh_hard + self.easy_neg_ratio
        # anything above _thresh_easy → cross-document

        logger.info(
            "📊 Pair ratios – positive: %.2f | hard_neg: %.2f | easy_neg: %.2f | cross_doc: %.2f",
            self.positive_ratio,
            self.hard_neg_ratio,
            self.easy_neg_ratio,
            self.cross_doc_ratio,
        )

    # ------------------------------------------------------------------
    # Strip extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_width(image_width: int) -> int:
        """Compute strip width: ``max(32, image_width // 20)``."""
        return max(32, image_width // 20)

    @staticmethod
    def _cut_strip(img: np.ndarray, x: int, width: int) -> np.ndarray:
        """Cut a vertical strip from *img* starting at column *x*.

        The strip is clamped to the image bounds.
        """
        h, w = img.shape[:2]
        x = max(0, min(x, w - 1))
        x_end = min(x + width, w)
        return img[:, x:x_end]

    # ------------------------------------------------------------------
    # Pair generators
    # ------------------------------------------------------------------

    def _make_positive(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Two adjacent strips → label 1."""
        h, w = img.shape[:2]
        sw = self._strip_width(w)
        max_x = w - 2 * sw
        if max_x < 0:
            # Image too narrow: use full width halves
            mid = w // 2
            left = img[:, :mid]
            right = img[:, mid : mid + mid]
            pair = np.concatenate([left, right], axis=1)
            return pair, 1.0
        x = random.randint(0, max_x)
        left = self._cut_strip(img, x, sw)
        right = self._cut_strip(img, x + sw, sw)
        pair = np.concatenate([left, right], axis=1)
        return pair, 1.0

    def _make_hard_negative(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Adjacent strips but the right strip is shifted vertically → label 0."""
        h, w = img.shape[:2]
        sw = self._strip_width(w)
        max_x = w - 2 * sw
        if max_x < 0:
            max_x = 0
        x = random.randint(0, max(0, max_x))
        left = self._cut_strip(img, x, sw)
        right_raw = self._cut_strip(img, x + sw, sw)

        shift = random.choice([10, 20, 30]) * random.choice([-1, 1])

        # Place right strip on a white canvas with vertical shift
        canvas = np.full_like(right_raw, 255)
        rh, rw = right_raw.shape[:2]
        if shift > 0:
            src_end = max(0, rh - shift)
            canvas[shift : shift + src_end] = right_raw[:src_end]
        else:
            abs_shift = abs(shift)
            src_start = min(abs_shift, rh)
            remaining = rh - src_start
            canvas[:remaining] = right_raw[src_start : src_start + remaining]

        # Make sure left and canvas have the same width
        min_w = min(left.shape[1], canvas.shape[1])
        left = left[:, :min_w]
        canvas = canvas[:, :min_w]

        pair = np.concatenate([left, canvas], axis=1)
        return pair, 0.0

    def _make_easy_negative(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Two non-adjacent strips from the same image → label 0."""
        h, w = img.shape[:2]
        sw = self._strip_width(w)
        if w < 3 * sw:
            # Fallback: just take first and last columns
            left = img[:, :sw]
            right = img[:, max(0, w - sw) :]
            # Ensure same width
            min_w = min(left.shape[1], right.shape[1])
            pair = np.concatenate([left[:, :min_w], right[:, :min_w]], axis=1)
            return pair, 0.0

        x1 = random.randint(0, w - sw)
        # Ensure x2 is not adjacent (at least 2*sw apart)
        attempts = 0
        x2 = x1
        while attempts < 20:
            x2 = random.randint(0, w - sw)
            if abs(x2 - x1) >= 2 * sw:
                break
            attempts += 1
        # If we couldn't find a good one, just pick the opposite end
        if abs(x2 - x1) < 2 * sw:
            x2 = 0 if x1 > w // 2 else w - sw

        left = self._cut_strip(img, x1, sw)
        right = self._cut_strip(img, x2, sw)

        min_w = min(left.shape[1], right.shape[1])
        pair = np.concatenate([left[:, :min_w], right[:, :min_w]], axis=1)
        return pair, 0.0

    def _make_cross_document(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """One strip from each of two *different* images → label 0."""
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        sw_a = self._strip_width(w_a)
        sw_b = self._strip_width(w_b)

        x_a = random.randint(0, max(0, w_a - sw_a))
        x_b = random.randint(0, max(0, w_b - sw_b))

        left = self._cut_strip(img_a, x_a, sw_a)
        right = self._cut_strip(img_b, x_b, sw_b)

        # Match heights – pad shorter strip with white
        target_h = max(left.shape[0], right.shape[0])
        if left.shape[0] < target_h:
            pad = np.full(
                (target_h - left.shape[0], left.shape[1], left.shape[2]),
                255,
                dtype=left.dtype,
            )
            left = np.concatenate([left, pad], axis=0)
        if right.shape[0] < target_h:
            pad = np.full(
                (target_h - right.shape[0], right.shape[1], right.shape[2]),
                255,
                dtype=right.dtype,
            )
            right = np.concatenate([right, pad], axis=0)

        pair = np.concatenate([left, right], axis=1)
        return pair, 0.0

    # ------------------------------------------------------------------
    # Main iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(image_tensor, label_tensor)`` pairs indefinitely."""
        stream_rng = random.Random()
        buffer: list[np.ndarray] = []
        warmup_target = min(50, max(1, int(self.buffer_size * 0.05)))

        # Determine worker info for shard splitting
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        logger.info(
            "🚀 Starting StreamingShredDataset (split=%s, buffer=%d, warmup=%d, worker=%d/%d)",
            self.split,
            self.buffer_size,
            warmup_target,
            worker_id,
            num_workers,
        )

        ds = load_dataset(
            self.dataset_path,
            split=self.split,
            streaming=self.streaming,
        )

        if not self.streaming:
            ds = ds.to_iterable_dataset()

        # Shard the stream across workers so each downloads a distinct slice
        if num_workers > 1:
            ds = ds.shard(num_shards=num_workers, index=worker_id)

        for sample in ds:
            # ---- extract & preprocess image ---------------------------
            raw_img: Image.Image | None = sample.get("image")
            if raw_img is None:
                continue
            try:
                arr = np.array(raw_img.convert("RGB"))
            except Exception:
                logger.debug("⏭️  Skipped unreadable image.")
                continue

            if arr.size == 0 or arr.shape[0] < 16 or arr.shape[1] < 16:
                continue

            arr = _ensure_rgb(arr)
            arr = _resize_to_max_width(arr, self.max_image_width)

            # ---- buffer management ------------------------------------
            if len(buffer) >= self.buffer_size:
                evict_idx = random.randint(0, len(buffer) - 1)
                buffer[evict_idx] = arr
            else:
                buffer.append(arr)

            # ---- warmup phase -----------------------------------------
            if len(buffer) < warmup_target:
                if len(buffer) % max(1, warmup_target // 5) == 0:
                    logger.info(
                        "⏳ Warming up buffer: %d / %d", len(buffer), warmup_target
                    )
                continue

            # ---- select a primary image from the buffer ---------------
            idx = random.randint(0, len(buffer) - 1)
            img = buffer[idx]

            # Occasional eviction to keep buffer fresh (10% chance)
            if random.random() < 0.10 and len(buffer) > warmup_target:
                buffer.pop(idx)

            # ---- choose pair type via cumulative thresholds -----------
            roll = random.random()
            if roll < self._thresh_pos:
                pair_type = self._PAIR_POSITIVE
            elif roll < self._thresh_hard:
                pair_type = self._PAIR_HARD_NEG
            elif roll < self._thresh_easy:
                pair_type = self._PAIR_EASY_NEG
            else:
                pair_type = self._PAIR_CROSS_DOC

            # ---- generate pair ----------------------------------------
            try:
                if pair_type == self._PAIR_POSITIVE:
                    pair_arr, label = self._make_positive(img)

                elif pair_type == self._PAIR_HARD_NEG:
                    pair_arr, label = self._make_hard_negative(img)

                elif pair_type == self._PAIR_EASY_NEG:
                    pair_arr, label = self._make_easy_negative(img)

                else:  # cross-document
                    if len(buffer) < 2:
                        # Not enough images – fall back to easy negative
                        pair_arr, label = self._make_easy_negative(img)
                    else:
                        other_idx = idx
                        safety = 0
                        while other_idx == idx and safety < 10:
                            other_idx = random.randint(0, len(buffer) - 1)
                            safety += 1
                        if other_idx == idx:
                            # Very unlikely but handle gracefully
                            pair_arr, label = self._make_easy_negative(img)
                        else:
                            pair_arr, label = self._make_cross_document(
                                img, buffer[other_idx]
                            )
            except Exception as exc:
                logger.warning("⚠️  Pair generation failed (%s): %s", pair_type, exc)
                continue

            # ---- convert to tensor & yield ----------------------------
            if pair_arr.size == 0:
                continue

            pil_pair = Image.fromarray(pair_arr.astype(np.uint8))
            if self.transform is not None:
                tensor = self.transform(pil_pair)
            else:
                tensor = transforms.ToTensor()(pil_pair)

            yield tensor, torch.tensor(label, dtype=torch.float32)
