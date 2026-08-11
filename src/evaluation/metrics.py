"""Multi-page reconstruction evaluation metrics.

Provides comprehensive metrics for evaluating both page clustering quality
and within-page ordering accuracy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusteringMetrics:
    """Metrics for evaluating page clustering quality."""

    adjusted_rand_index: float = 0.0
    normalized_mutual_info: float = 0.0
    num_predicted_pages: int = 0
    num_true_pages: int = 0
    num_noise_strips: int = 0
    inter_page_error_rate: float = 0.0

    def __str__(self) -> str:
        return (
            f"Clustering Metrics:\n"
            f"  ARI:                {self.adjusted_rand_index:.4f}\n"
            f"  NMI:                {self.normalized_mutual_info:.4f}\n"
            f"  Predicted pages:    {self.num_predicted_pages}\n"
            f"  True pages:         {self.num_true_pages}\n"
            f"  Noise strips:       {self.num_noise_strips}\n"
            f"  Inter-page errors:  {self.inter_page_error_rate:.2%}"
        )


@dataclass
class OrderingMetrics:
    """Metrics for evaluating within-page strip ordering quality."""

    pairwise_accuracy: float = 0.0
    perfect_page_rate: float = 0.0
    mean_kendall_tau: float = 0.0
    mean_displacement: float = 0.0

    def __str__(self) -> str:
        return (
            f"Ordering Metrics:\n"
            f"  Pairwise accuracy:  {self.pairwise_accuracy:.2%}\n"
            f"  Perfect page rate:  {self.perfect_page_rate:.2%}\n"
            f"  Mean Kendall's τ:   {self.mean_kendall_tau:.4f}\n"
            f"  Mean displacement:  {self.mean_displacement:.2f}"
        )


@dataclass
class ReconstructionMetrics:
    """Combined metrics for full multi-page reconstruction."""

    clustering: ClusteringMetrics = field(default_factory=ClusteringMetrics)
    ordering: OrderingMetrics = field(default_factory=OrderingMetrics)

    def __str__(self) -> str:
        return f"{self.clustering}\n\n{self.ordering}"


def compute_clustering_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> ClusteringMetrics:
    """Compute clustering quality metrics.

    Args:
        true_labels: Ground truth page labels for each strip, shape (n,).
        pred_labels: Predicted page labels for each strip, shape (n,).
            Noise strips should be labeled -1.

    Returns:
        ClusteringMetrics with ARI, NMI, and page counts.
    """
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for clustering metrics.")

    metrics = ClusteringMetrics()
    metrics.num_true_pages = len(set(true_labels))
    metrics.num_noise_strips = int(np.sum(pred_labels == -1))

    # For ARI/NMI, exclude noise strips
    non_noise_mask = pred_labels != -1
    if non_noise_mask.sum() > 0:
        true_non_noise = true_labels[non_noise_mask]
        pred_non_noise = pred_labels[non_noise_mask]
        metrics.adjusted_rand_index = adjusted_rand_score(
            true_non_noise, pred_non_noise
        )
        metrics.normalized_mutual_info = normalized_mutual_info_score(
            true_non_noise, pred_non_noise
        )
        metrics.num_predicted_pages = len(set(pred_non_noise))
    else:
        metrics.num_predicted_pages = 0

    return metrics


def compute_pairwise_accuracy(
    true_order: list[int],
    pred_order: list[int],
) -> float:
    """Compute pairwise neighbor accuracy.

    For each consecutive pair in the predicted order, check if they are
    truly adjacent in the ground truth.

    Args:
        true_order: Ground truth strip indices in correct order.
        pred_order: Predicted strip indices in predicted order.

    Returns:
        Fraction of consecutive pairs that are correct.
    """
    if len(pred_order) < 2:
        return 0.0

    # Build true adjacency set
    true_pairs = set()
    for k in range(len(true_order) - 1):
        true_pairs.add((true_order[k], true_order[k + 1]))

    correct = 0
    total = len(pred_order) - 1
    for k in range(total):
        if (pred_order[k], pred_order[k + 1]) in true_pairs:
            correct += 1

    return correct / total if total > 0 else 0.0


def compute_kendall_tau(
    true_order: list[int],
    pred_order: list[int],
) -> float:
    """Compute Kendall's Tau rank correlation between two orderings.

    Args:
        true_order: Ground truth strip indices in correct order.
        pred_order: Predicted strip indices in predicted order.

    Returns:
        Kendall's Tau correlation coefficient in [-1, 1].
    """
    try:
        from scipy.stats import kendalltau
    except ImportError:
        raise ImportError("scipy is required for Kendall's Tau.")

    if len(true_order) < 2 or len(pred_order) < 2:
        return 0.0

    # Convert to rank arrays
    true_set = set(true_order)
    pred_set = set(pred_order)
    common = sorted(true_set & pred_set)

    if len(common) < 2:
        return 0.0

    true_ranks = {v: i for i, v in enumerate(true_order)}
    pred_ranks = {v: i for i, v in enumerate(pred_order)}

    true_r = [true_ranks[v] for v in common]
    pred_r = [pred_ranks[v] for v in common]

    tau, _ = kendalltau(true_r, pred_r)
    return float(tau) if not np.isnan(tau) else 0.0


def compute_mean_displacement(
    true_order: list[int],
    pred_order: list[int],
) -> float:
    """Compute mean displacement of strips from their true positions.

    For each strip, measures |predicted_position - true_position|.

    Args:
        true_order: Ground truth strip indices in correct order.
        pred_order: Predicted strip indices in predicted order.

    Returns:
        Average absolute displacement.
    """
    if not true_order or not pred_order:
        return 0.0

    true_pos = {v: i for i, v in enumerate(true_order)}
    pred_pos = {v: i for i, v in enumerate(pred_order)}

    common = set(true_pos.keys()) & set(pred_pos.keys())
    if not common:
        return 0.0

    displacements = [abs(pred_pos[v] - true_pos[v]) for v in common]
    return sum(displacements) / len(displacements)


def compute_inter_page_error_rate(
    pred_pages: list[list[int]],
    true_page_labels: dict[int, int],
) -> float:
    """Compute what fraction of predicted consecutive pairs cross page boundaries.

    Args:
        pred_pages: List of predicted pages, each a list of strip indices.
        true_page_labels: Dict mapping strip index → true page label.

    Returns:
        Fraction of consecutive pairs that span different true pages.
    """
    total_pairs = 0
    cross_page_pairs = 0

    for page in pred_pages:
        for k in range(len(page) - 1):
            total_pairs += 1
            s_a, s_b = page[k], page[k + 1]
            if s_a in true_page_labels and s_b in true_page_labels:
                if true_page_labels[s_a] != true_page_labels[s_b]:
                    cross_page_pairs += 1

    return cross_page_pairs / total_pairs if total_pairs > 0 else 0.0


def evaluate_reconstruction(
    pred_pages: list[list[int]],
    true_pages: dict[int, list[int]],
    true_page_labels: dict[int, int],
    pred_page_labels: np.ndarray | None = None,
    true_label_array: np.ndarray | None = None,
) -> ReconstructionMetrics:
    """Full evaluation of a multi-page reconstruction.

    Args:
        pred_pages: List of predicted pages, each a list of strip indices.
        true_pages: Dict mapping true_page_id → list of strip indices in order.
        true_page_labels: Dict mapping strip_index → true page label.
        pred_page_labels: Array of predicted cluster labels (for clustering metrics).
        true_label_array: Array of true page labels (for clustering metrics).

    Returns:
        ReconstructionMetrics with both clustering and ordering scores.
    """
    metrics = ReconstructionMetrics()

    # --- Clustering Metrics ---
    if pred_page_labels is not None and true_label_array is not None:
        metrics.clustering = compute_clustering_metrics(
            true_label_array, pred_page_labels
        )

    metrics.clustering.inter_page_error_rate = compute_inter_page_error_rate(
        pred_pages, true_page_labels
    )
    metrics.clustering.num_predicted_pages = len(pred_pages)
    metrics.clustering.num_true_pages = len(true_pages)

    # --- Ordering Metrics ---
    # Match predicted pages to true pages (best overlap)
    pairwise_accs = []
    kendall_taus = []
    displacements = []
    perfect_pages = 0

    for pred_page in pred_pages:
        if len(pred_page) < 2:
            continue

        # Find the best matching true page (by overlap)
        best_true_page = None
        best_overlap = 0
        for true_id, true_order in true_pages.items():
            overlap = len(set(pred_page) & set(true_order))
            if overlap > best_overlap:
                best_overlap = overlap
                best_true_page = true_order

        if best_true_page is None:
            continue

        # Compute ordering metrics against the best match
        pa = compute_pairwise_accuracy(best_true_page, pred_page)
        kt = compute_kendall_tau(best_true_page, pred_page)
        md = compute_mean_displacement(best_true_page, pred_page)

        pairwise_accs.append(pa)
        kendall_taus.append(kt)
        displacements.append(md)

        # Check if perfect (all strips correct, in order)
        if set(pred_page) == set(best_true_page) and pred_page == best_true_page:
            perfect_pages += 1

    if pairwise_accs:
        metrics.ordering.pairwise_accuracy = float(np.mean(pairwise_accs))
        metrics.ordering.mean_kendall_tau = float(np.mean(kendall_taus))
        metrics.ordering.mean_displacement = float(np.mean(displacements))
        metrics.ordering.perfect_page_rate = (
            perfect_pages / len(true_pages) if true_pages else 0.0
        )

    return metrics
