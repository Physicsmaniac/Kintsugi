"""Asymmetric Travelling Salesman Problem (ATSP) solver for strip ordering.

Converts the n×n score matrix into a cost matrix and solves the ATSP to
recover the optimal left-to-right ordering of strips within a single page.

Strategy:
    1. ``C[i,j] = 1 − score[i,j]``  (lower cost ⇒ strip j more likely right of i)
    2. Append a dummy node *d* with ``C[d,j] = C[i,d] = 0`` to convert the
       open Hamiltonian-path problem into a closed TSP tour.
    3. Solve ATSP via simulated annealing (``python-tsp``).
    4. Remove the dummy node and read off the path order.
"""

from __future__ import annotations

import itertools
import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-import python-tsp so the module can still be imported if it is missing.
_PYTHON_TSP_AVAILABLE: bool = False
try:
    from python_tsp.heuristics import solve_tsp_simulated_annealing  # type: ignore[import-untyped]

    _PYTHON_TSP_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Cost matrix construction
# ---------------------------------------------------------------------------

def build_cost_matrix(
    scores: np.ndarray,
    temperature: float = 0.5,
    use_logits: bool = False,
) -> np.ndarray:
    """Convert a score/logit matrix to a cost matrix for the ATSP.

    Parameters
    ----------
    scores : np.ndarray
        n×n matrix.  When *use_logits* is ``False`` these are sigmoid
        probabilities in [0, 1]; when ``True`` they are raw logits.
    temperature : float
        Softmax temperature applied to logits before conversion.
        Only used when ``use_logits=True``.
    use_logits : bool
        If ``True``, apply ``sigmoid(logit / temperature)`` to obtain
        probabilities first.

    Returns
    -------
    cost_matrix : np.ndarray
        n×n cost matrix where ``cost[i,j] = 1 − prob[i,j]``.
    """
    if use_logits:
        # Temperature-scaled sigmoid
        scaled = scores / max(temperature, 1e-8)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -50, 50)))
    else:
        probs = np.asarray(scores, dtype=np.float64)

    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    costs = -np.log(probs)
    # Zero out diagonal (self-adjacency is meaningless)
    np.fill_diagonal(costs, 0.0)
    return costs


def _add_dummy_node(cost_matrix: np.ndarray) -> np.ndarray:
    """Append a zero-cost dummy node to turn the open path into a tour.

    The dummy connects to all real nodes with zero cost so that
    "breaking" the tour at the dummy recovers the cheapest Hamiltonian
    path.

    Parameters
    ----------
    cost_matrix : np.ndarray
        n×n cost matrix.

    Returns
    -------
    augmented : np.ndarray
        (n+1)×(n+1) cost matrix with the dummy as the last row/column.
    """
    n = cost_matrix.shape[0]
    augmented = np.zeros((n + 1, n + 1), dtype=np.float64)
    augmented[:n, :n] = cost_matrix
    # Row n (dummy → all nodes) and column n (all nodes → dummy) stay 0
    return augmented


def _remove_dummy_from_tour(tour: Sequence[int], n_real: int) -> list[int]:
    """Remove the dummy node from a closed tour to recover the open path.

    Parameters
    ----------
    tour : sequence of int
        Closed tour of length n+1 (indices 0…n, where n is the dummy).
    n_real : int
        Number of real strip nodes.

    Returns
    -------
    path : list[int]
        Ordered strip indices forming the Hamiltonian path.
    """
    tour = list(tour)
    dummy = n_real  # dummy node index

    if dummy not in tour:
        logger.warning("Dummy node %d not found in tour; returning tour as-is.", dummy)
        return [x for x in tour if x != dummy]

    idx = tour.index(dummy)
    # Rotate so that the element *after* dummy is first
    rotated = tour[idx + 1:] + tour[:idx]
    # Filter out any accidental duplicates of dummy
    return [x for x in rotated if x != dummy]


# ---------------------------------------------------------------------------
# ATSP solver (simulated annealing via python-tsp)
# ---------------------------------------------------------------------------

def solve_atsp(
    score_matrix: np.ndarray,
    temperature: float = 0.5,
    use_logits: bool = False,
) -> list[int]:
    """Solve the strip ordering as an Asymmetric TSP.

    Parameters
    ----------
    score_matrix : np.ndarray
        n×n matrix of pairwise scores (probabilities or logits).
    temperature : float
        Temperature for logit-to-probability conversion.
    use_logits : bool
        Whether ``score_matrix`` contains raw logits.

    Returns
    -------
    path : list[int]
        Ordered strip indices forming the best Hamiltonian path found.

    Raises
    ------
    ImportError
        If ``python-tsp`` is not installed.
    """
    if not _PYTHON_TSP_AVAILABLE:
        raise ImportError(
            "python-tsp is required for the ATSP solver.  "
            "Install it with:  pip install python-tsp"
        )

    n = score_matrix.shape[0]
    if n <= 1:
        return list(range(n))

    # For very small instances, exact brute-force is cheap
    if n <= 8:
        logger.info("n=%d ≤ 8 → using exact brute-force solver.", n)
        return solve_bruteforce(score_matrix, temperature=temperature, use_logits=use_logits)

    cost = build_cost_matrix(score_matrix, temperature=temperature, use_logits=use_logits)
    augmented = _add_dummy_node(cost)

    logger.info("Solving ATSP via simulated annealing (n=%d + dummy)…", n)
    tour, total_cost = solve_tsp_simulated_annealing(augmented)
    logger.info("SA finished.  Total tour cost: %.4f", total_cost)

    path = _remove_dummy_from_tour(tour, n_real=n)
    logger.info("Recovered path (length=%d): %s", len(path), path)
    return path


# ---------------------------------------------------------------------------
# Brute-force exact solver for small n
# ---------------------------------------------------------------------------

def solve_bruteforce(
    score_matrix: np.ndarray,
    temperature: float = 0.5,
    use_logits: bool = False,
) -> list[int]:
    """Exact Hamiltonian-path solver via brute-force permutation enumeration.

    Only practical for ``n ≤ 8`` (8! = 40 320 permutations).

    Parameters
    ----------
    score_matrix : np.ndarray
        n×n score matrix.
    temperature : float
        Temperature for logit conversion (see :func:`build_cost_matrix`).
    use_logits : bool
        Whether ``score_matrix`` contains raw logits.

    Returns
    -------
    best_path : list[int]
        Globally optimal strip ordering.
    """
    n = score_matrix.shape[0]
    if n <= 1:
        return list(range(n))

    cost = build_cost_matrix(score_matrix, temperature=temperature, use_logits=use_logits)

    best_cost = float("inf")
    best_perm: tuple[int, ...] = tuple(range(n))

    for perm in itertools.permutations(range(n)):
        total = 0.0
        for k in range(len(perm) - 1):
            total += cost[perm[k], perm[k + 1]]
        if total < best_cost:
            best_cost = total
            best_perm = perm

    logger.info(
        "Brute-force (n=%d): best cost=%.4f, path=%s",
        n, best_cost, list(best_perm),
    )
    return list(best_perm)


# ---------------------------------------------------------------------------
# Convenience: solve within pre-defined clusters
# ---------------------------------------------------------------------------

def solve_atsp_with_clusters(
    score_matrix: np.ndarray,
    clusters: dict[int, list[int]],
    temperature: float = 0.5,
    use_logits: bool = False,
) -> dict[int, list[int]]:
    """Run the ATSP solver independently within each cluster.

    Parameters
    ----------
    score_matrix : np.ndarray
        Full n×n score matrix.
    clusters : dict[int, list[int]]
        Cluster label → list of strip indices.
    temperature : float
        Temperature for cost conversion.
    use_logits : bool
        Whether ``score_matrix`` contains raw logits.

    Returns
    -------
    ordered : dict[int, list[int]]
        Same keys, with strip indices reordered by the ATSP solution.
    """
    ordered: dict[int, list[int]] = {}

    for label, indices in clusters.items():
        if len(indices) <= 1:
            ordered[label] = list(indices)
            continue

        # Build sub-matrix
        sub_n = len(indices)
        sub_matrix = np.zeros((sub_n, sub_n), dtype=score_matrix.dtype)
        for li, oi in enumerate(indices):
            for lj, oj in enumerate(indices):
                sub_matrix[li, lj] = score_matrix[oi, oj]

        # Solve
        local_path = solve_atsp(
            sub_matrix, temperature=temperature, use_logits=use_logits,
        )
        ordered[label] = [indices[k] for k in local_path]

    return ordered
