"""Solver package for multi-page shredded document reconstruction.

Submodules
----------
scoring
    Pairwise strip-adjacency scoring (score matrix computation).
greedy
    Baseline greedy chain-building solver.
atsp
    Asymmetric TSP solver for optimal strip ordering.
clustering
    HDBSCAN-based page clustering with noise reassignment and
    spectral block-split detection.
"""

from __future__ import annotations

from .atsp import solve_atsp
from .clustering import cluster_strips_by_page, detect_and_split_merged_clusters
from .greedy import solve_greedy
from .scoring import compute_score_matrix, compute_score_matrix_batched

__all__ = [
    "compute_score_matrix",
    "compute_score_matrix_batched",
    "solve_greedy",
    "solve_atsp",
    "cluster_strips_by_page",
    "detect_and_split_merged_clusters",
]
