from __future__ import annotations

"""Streaming shredded-document pair dataset for seam classification.

Streams document images from HuggingFace ``chainyo/rvl-cdip``, maintains a
rolling buffer, and yields horizontally-concatenated strip pairs with labels:

* **Positive (1)** – two adjacent vertical strips.
* **Hard negative (0)** – adjacent strips with a vertical pixel shift.
* **Easy negative (0)** – two non-adjacent strips from the *same* image.
* **Cross-document negative (0)** – strips from *different* images in the buffer.
* **Seam-adjacent cross-doc (0)** – seam edges from *different* images (hardest).

Key improvements over the original dataset:
    * Realistic strip counts (10-15 per image) matching inference conditions.
    * Seam-edge cropping: takes only the boundary pixels, preserving resolution.
    * Adjusted positive ratio (15%) matching the low positive rate at inference.
    * New seam-adjacent cross-doc negative type for the hardest failure mode.

Ratios are fully configurable via the constructor.
"""

import logging
import random
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
import PIL
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

logger = logging.getLogger(__name__)

# Suppress noisy HTTP probe logs from HuggingFace internals
for _noisy_logger in ("httpx", "huggingface_hub"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Seam edge width — how many pixels to take from each strip's boundary.
# Must match TRAINING_STRIP_WIDTH used in preprocessing/scoring.
SEAM_EDGE_WIDTH = 48

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
    seam_cross_doc_ratio:
        Fraction of yielded pairs that are seam-adjacent cross-doc negatives.
    num_strips_per_image:
        Number of strips to cut each image into (10-15 is realistic).
    max_image_width:
        Images wider than this are down-scaled before strip extraction.
    dataset_path:
        HuggingFace dataset identifier.
    """

    _PAIR_POSITIVE = "positive"
    _PAIR_HARD_NEG = "hard_negative"
    _PAIR_EASY_NEG = "easy_negative"
    _PAIR_CROSS_DOC = "cross_document"
    _PAIR_SEAM_CROSS_DOC = "seam_cross_document"

    def __init__(
        self,
        split: str = "train",
        transform: transforms.Compose | None = None,
        buffer_size: int = 1000,
        positive_ratio: float = 0.15,
        hard_neg_ratio: float = 0.25,
        easy_neg_ratio: float = 0.20,
        cross_doc_ratio: float = 0.20,
        seam_cross_doc_ratio: float = 0.20,
        num_strips_per_image: int = 12,
        max_image_width: int = 800,
        dataset_path: str = "chainyo/rvl-cdip",
        streaming: bool = True,
    ) -> None:
        super().__init__()
        self.split = split
        self.transform = transform
        self.buffer_size = max(1, buffer_size)
        self.num_strips_per_image = max(4, num_strips_per_image)
        self.max_image_width = max(64, max_image_width)
        self.dataset_path = dataset_path
        self.streaming = streaming
        self._local_ds = None

        # Load local datasets once in the main process to avoid file lock deadlocks
        # when multiple PyTorch workers spawn simultaneously.
        if not self.streaming:
            from datasets import load_dataset
            self._local_ds = load_dataset(self.dataset_path, split=self.split, streaming=False)

        # Normalise ratios so they sum to 1.0 --------------------------
        total = (positive_ratio + hard_neg_ratio + easy_neg_ratio
                 + cross_doc_ratio + seam_cross_doc_ratio)
        if total <= 0:
            logger.warning(
                "⚠️  All pair ratios are zero – falling back to uniform distribution."
            )
            total = 5.0
            positive_ratio = hard_neg_ratio = easy_neg_ratio = 1.0
            cross_doc_ratio = seam_cross_doc_ratio = 1.0
        self.positive_ratio = positive_ratio / total
        self.hard_neg_ratio = hard_neg_ratio / total
        self.easy_neg_ratio = easy_neg_ratio / total
        self.cross_doc_ratio = cross_doc_ratio / total
        self.seam_cross_doc_ratio = seam_cross_doc_ratio / total

        # Cumulative thresholds for random selection --------------------
        self._thresh_pos = self.positive_ratio
        self._thresh_hard = self._thresh_pos + self.hard_neg_ratio
        self._thresh_easy = self._thresh_hard + self.easy_neg_ratio
        self._thresh_cross = self._thresh_easy + self.cross_doc_ratio
        # anything above _thresh_cross → seam cross-document

        logger.info(
            "📊 Pair ratios – pos: %.2f | hard_neg: %.2f | easy_neg: %.2f | "
            "cross_doc: %.2f | seam_cross_doc: %.2f",
            self.positive_ratio,
            self.hard_neg_ratio,
            self.easy_neg_ratio,
            self.cross_doc_ratio,
            self.seam_cross_doc_ratio,
        )

    # ------------------------------------------------------------------
    # Strip extraction
    # ------------------------------------------------------------------

    def _shred_image(self, img: np.ndarray) -> list[np.ndarray]:
        """Cut an image into ``num_strips_per_image`` vertical strips.

        Returns a list of strips. Each strip has the full image height
        and approximately ``image_width / num_strips`` width.
        """
        h, w = img.shape[:2]
        n = self.num_strips_per_image
        sw = max(8, w // n)
        strips = []
        for i in range(n):
            x_start = i * sw
            x_end = min((i + 1) * sw, w)
            if x_end <= x_start:
                break
            strips.append(img[:, x_start:x_end])
        return strips

    @staticmethod
    def _get_seam_edges(
        left_strip: np.ndarray,
        right_strip: np.ndarray,
        edge_width: int = SEAM_EDGE_WIDTH,
    ) -> np.ndarray:
        """Extract and concatenate the seam boundary pixels from two strips.

        Takes the rightmost ``edge_width`` pixels from the left strip and
        the leftmost ``edge_width`` pixels from the right strip, then
        concatenates them horizontally.

        Parameters
        ----------
        left_strip:
            HxW1xC array (the left strip).
        right_strip:
            HxW2xC array (the right strip).
        edge_width:
            Pixels to take from each side of the seam.

        Returns
        -------
        np.ndarray
            Hx(2*edge_width)xC seam pair image.
        """
        h_l, w_l = left_strip.shape[:2]
        h_r, w_r = right_strip.shape[:2]

        # Take right edge of left strip
        if w_l >= edge_width:
            left_edge = left_strip[:, w_l - edge_width:]
        else:
            # Pad with white if strip is narrower than edge_width
            pad = np.full((h_l, edge_width - w_l, left_strip.shape[2]), 255, dtype=left_strip.dtype)
            left_edge = np.concatenate([pad, left_strip], axis=1)

        # Take left edge of right strip
        if w_r >= edge_width:
            right_edge = right_strip[:, :edge_width]
        else:
            pad = np.full((h_r, edge_width - w_r, right_strip.shape[2]), 255, dtype=right_strip.dtype)
            right_edge = np.concatenate([right_strip, pad], axis=1)

        # Match heights
        min_h = min(left_edge.shape[0], right_edge.shape[0])
        left_edge = left_edge[:min_h]
        right_edge = right_edge[:min_h]

        return np.concatenate([left_edge, right_edge], axis=1)

    # ------------------------------------------------------------------
    # Pair generators
    # ------------------------------------------------------------------

    def _make_positive(self, strips: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Two adjacent strips → label 1. Uses seam-edge cropping."""
        if len(strips) < 2:
            return self._make_positive_fallback(strips[0] if strips else np.zeros((16, 16, 3), dtype=np.uint8))

        idx = random.randint(0, len(strips) - 2)
        pair = self._get_seam_edges(strips[idx], strips[idx + 1])
        return pair, 1.0

    def _make_positive_fallback(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Fallback: cut image in half for a positive pair."""
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, max(0, mid - SEAM_EDGE_WIDTH):mid]
        right = img[:, mid:min(w, mid + SEAM_EDGE_WIDTH)]
        # Pad if needed
        if left.shape[1] < SEAM_EDGE_WIDTH:
            pad = np.full((h, SEAM_EDGE_WIDTH - left.shape[1], 3), 255, dtype=np.uint8)
            left = np.concatenate([pad, left], axis=1)
        if right.shape[1] < SEAM_EDGE_WIDTH:
            pad = np.full((h, SEAM_EDGE_WIDTH - right.shape[1], 3), 255, dtype=np.uint8)
            right = np.concatenate([right, pad], axis=1)
        pair = np.concatenate([left, right], axis=1)
        return pair, 1.0

    def _make_hard_negative(self, strips: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Adjacent strips but the right strip is shifted vertically → label 0."""
        if len(strips) < 2:
            return self._make_easy_negative_from_image(strips[0] if strips else np.zeros((16, 16, 3), dtype=np.uint8))

        idx = random.randint(0, len(strips) - 2)
        left_strip = strips[idx]
        right_strip = strips[idx + 1]

        # Apply vertical shift to right strip
        shift = random.choice([10, 20, 30]) * random.choice([-1, 1])
        h, w = right_strip.shape[:2]
        canvas = np.full_like(right_strip, 255)
        if shift > 0:
            src_end = max(0, h - shift)
            canvas[shift:shift + src_end] = right_strip[:src_end]
        else:
            abs_shift = abs(shift)
            src_start = min(abs_shift, h)
            remaining = h - src_start
            canvas[:remaining] = right_strip[src_start:src_start + remaining]

        pair = self._get_seam_edges(left_strip, canvas)
        return pair, 0.0

    def _make_easy_negative(self, strips: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Two non-adjacent strips from the same image → label 0."""
        if len(strips) < 3:
            return self._make_easy_negative_from_image(
                np.concatenate(strips, axis=1) if strips else np.zeros((16, 16, 3), dtype=np.uint8))

        # Pick two strips that are at least 2 apart
        attempts = 0
        while attempts < 20:
            i = random.randint(0, len(strips) - 1)
            j = random.randint(0, len(strips) - 1)
            if abs(i - j) >= 2:
                break
            attempts += 1
        if abs(i - j) < 2:
            i, j = 0, len(strips) - 1

        pair = self._get_seam_edges(strips[i], strips[j])
        return pair, 0.0

    def _make_easy_negative_from_image(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Fallback easy negative from a single image."""
        h, w = img.shape[:2]
        ew = SEAM_EDGE_WIDTH
        if w < 3 * ew:
            left = img[:, :ew] if w >= ew else np.full((h, ew, 3), 255, dtype=np.uint8)
            right = img[:, max(0, w - ew):] if w >= ew else np.full((h, ew, 3), 255, dtype=np.uint8)
            if right.shape[1] < ew:
                pad = np.full((h, ew - right.shape[1], 3), 255, dtype=np.uint8)
                right = np.concatenate([right, pad], axis=1)
            if left.shape[1] < ew:
                pad = np.full((h, ew - left.shape[1], 3), 255, dtype=np.uint8)
                left = np.concatenate([pad, left], axis=1)
        else:
            left = img[:, :ew]
            right = img[:, w - ew:]
        min_h = min(left.shape[0], right.shape[0])
        pair = np.concatenate([left[:min_h], right[:min_h]], axis=1)
        return pair, 0.0

    def _make_cross_document(
        self, strips_a: list[np.ndarray], strips_b: list[np.ndarray]
    ) -> tuple[np.ndarray, float]:
        """Random strip from each of two different images → label 0."""
        idx_a = random.randint(0, len(strips_a) - 1) if strips_a else 0
        idx_b = random.randint(0, len(strips_b) - 1) if strips_b else 0

        left = strips_a[idx_a] if strips_a else np.full((64, SEAM_EDGE_WIDTH, 3), 255, dtype=np.uint8)
        right = strips_b[idx_b] if strips_b else np.full((64, SEAM_EDGE_WIDTH, 3), 255, dtype=np.uint8)

        pair = self._get_seam_edges(left, right)
        return pair, 0.0

    def _make_seam_cross_document(
        self, strips_a: list[np.ndarray], strips_b: list[np.ndarray]
    ) -> tuple[np.ndarray, float]:
        """Seam-adjacent edges from two different images → label 0.

        This is the hardest negative type: takes the right edge of strip_a[i]
        (where it would join strip_a[i+1]) and the left edge of strip_b[j]
        (where strip_b[j-1] would join it). These edges look like plausible
        seam boundaries but come from different documents.
        """
        # Pick an interior strip boundary from each document
        if len(strips_a) >= 2:
            idx_a = random.randint(0, len(strips_a) - 2)
            left = strips_a[idx_a]  # Use right edge of this strip
        else:
            left = strips_a[0] if strips_a else np.full((64, SEAM_EDGE_WIDTH, 3), 255, dtype=np.uint8)

        if len(strips_b) >= 2:
            idx_b = random.randint(1, len(strips_b) - 1)
            right = strips_b[idx_b]  # Use left edge of this strip
        else:
            right = strips_b[0] if strips_b else np.full((64, SEAM_EDGE_WIDTH, 3), 255, dtype=np.uint8)

        pair = self._get_seam_edges(left, right)
        return pair, 0.0

    # ------------------------------------------------------------------
    # Main iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(image_tensor, label_tensor)`` pairs indefinitely."""
        stream_rng = random.Random()
        # Buffer stores pre-shredded strips for each image
        buffer: list[list[np.ndarray]] = []
        warmup_target = min(50, max(1, int(self.buffer_size * 0.05)))

        # Determine worker info for shard splitting
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        logger.info(
            "🚀 Starting StreamingShredDataset (split=%s, buffer=%d, warmup=%d, worker=%d/%d, "
            "strips_per_image=%d)",
            self.split,
            self.buffer_size,
            warmup_target,
            worker_id,
            num_workers,
            self.num_strips_per_image,
        )

        if self.streaming:
            from datasets import load_dataset
            ds = load_dataset(
                self.dataset_path,
                split=self.split,
                streaming=True,
            )
        else:
            ds = self._local_ds
            if num_workers > 1:
                ds = ds.shard(num_shards=num_workers, index=worker_id)

        # Wrap iteration to catch corrupt images that HuggingFace's decoder
        # fails to open (PIL.UnidentifiedImageError, OSError, etc.)
        def _safe_iter(dataset):
            it = iter(dataset)
            while True:
                try:
                    yield next(it)
                except StopIteration:
                    return
                except Exception as exc:
                    logger.debug("⏭️  Skipped corrupt image in dataset: %s", exc)
                    continue

        for sample in _safe_iter(ds):
            # ---- extract & preprocess image ---------------------------
            raw_img = sample.get("image")
            if raw_img is None:
                continue
            
            try:
                arr = np.array(raw_img.convert("RGB"))
            except Exception as exc:
                logger.debug("⏭️  Skipped unreadable image: %s", exc)
                continue

            if arr.size == 0 or arr.shape[0] < 16 or arr.shape[1] < 16:
                continue

            arr = _ensure_rgb(arr)
            arr = _resize_to_max_width(arr, self.max_image_width)

            # Shred into strips
            strips = self._shred_image(arr)
            if len(strips) < 2:
                continue

            # ---- buffer management ------------------------------------
            if len(buffer) >= self.buffer_size:
                evict_idx = random.randint(0, len(buffer) - 1)
                buffer[evict_idx] = strips
            else:
                buffer.append(strips)

            # ---- warmup phase -----------------------------------------
            if len(buffer) < warmup_target:
                if len(buffer) % max(1, warmup_target // 5) == 0:
                    logger.info(
                        "⏳ Warming up buffer: %d / %d", len(buffer), warmup_target
                    )
                continue

            # ---- select a primary image's strips from the buffer ------
            idx = random.randint(0, len(buffer) - 1)
            current_strips = buffer[idx]

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
            elif roll < self._thresh_cross:
                pair_type = self._PAIR_CROSS_DOC
            else:
                pair_type = self._PAIR_SEAM_CROSS_DOC

            # ---- generate pair ----------------------------------------
            try:
                if pair_type == self._PAIR_POSITIVE:
                    pair_arr, label = self._make_positive(current_strips)

                elif pair_type == self._PAIR_HARD_NEG:
                    pair_arr, label = self._make_hard_negative(current_strips)

                elif pair_type == self._PAIR_EASY_NEG:
                    pair_arr, label = self._make_easy_negative(current_strips)

                elif pair_type == self._PAIR_CROSS_DOC:
                    if len(buffer) < 2:
                        pair_arr, label = self._make_easy_negative(current_strips)
                    else:
                        other_idx = idx
                        safety = 0
                        while other_idx == idx and safety < 10:
                            other_idx = random.randint(0, len(buffer) - 1)
                            safety += 1
                        if other_idx == idx:
                            pair_arr, label = self._make_easy_negative(current_strips)
                        else:
                            pair_arr, label = self._make_cross_document(
                                current_strips, buffer[other_idx]
                            )

                else:  # seam cross-document
                    if len(buffer) < 2:
                        pair_arr, label = self._make_easy_negative(current_strips)
                    else:
                        other_idx = idx
                        safety = 0
                        while other_idx == idx and safety < 10:
                            other_idx = random.randint(0, len(buffer) - 1)
                            safety += 1
                        if other_idx == idx:
                            pair_arr, label = self._make_easy_negative(current_strips)
                        else:
                            pair_arr, label = self._make_seam_cross_document(
                                current_strips, buffer[other_idx]
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
