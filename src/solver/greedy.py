"""Greedy baseline solver for strip ordering.

Given an n×n score matrix where ``score[i, j]`` is the probability that
strip *j* belongs immediately to the right of strip *i*, this module
greedily builds chains (pages) by iteratively extending with the
highest-confidence unvisited neighbour.

Chain scoring heuristic: ``avg_edge_score × √len(chain)``
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _build_chain(
    start: int,
    score_matrix: np.ndarray,
    available: set[int],
    threshold: float,
) -> tuple[list[int], float]:
    """Build a single greedy chain starting from *start*.

    Parameters
    ----------
    start : int
        Index of the starting strip.
    score_matrix : np.ndarray
        n×n score matrix.
    available : set[int]
        Set of strip indices still available for assignment.
    threshold : float
        Minimum confidence to continue extending a chain.

    Returns
    -------
    chain : list[int]
        Ordered list of strip indices forming the chain.
    chain_score : float
        Heuristic chain score: ``avg_edge_score × √len(chain)``.
    """
    chain = [start]
    used = {start}
    edge_scores: list[float] = []

    while True:
        current = chain[-1]
        candidates = available - used
        if not candidates:
            break

        # Find highest-scoring right-neighbour among candidates
        best_idx = -1
        best_score = -1.0
        for c in candidates:
            s = float(score_matrix[current, c])
            if s > best_score:
                best_score = s
                best_idx = c

        if best_score < threshold or best_idx < 0:
            break

        chain.append(best_idx)
        used.add(best_idx)
        edge_scores.append(best_score)

    if len(edge_scores) > 0:
        avg_score = sum(edge_scores) / len(edge_scores)
    else:
        avg_score = 0.0

    chain_score = avg_score * math.sqrt(len(chain))
    return chain, chain_score


def solve_greedy(
    score_matrix: np.ndarray,
    threshold: float = 0.5,
    min_chain_length: int = 2,
) -> list[list[int]]:
    """Greedy strip-ordering solver.

    Iteratively builds chains by trying every available strip as a start
    node, greedily extending each, and committing the best-scoring chain
    as a page.  Remaining unassigned strips are returned as singletons.

    Parameters
    ----------
    score_matrix : np.ndarray
        n×n asymmetric score matrix.  ``score_matrix[i, j]`` is the
        probability that strip *j* is immediately right of strip *i*.
    threshold : float
        Minimum edge confidence to continue extending a chain.
    min_chain_length : int
        Chains shorter than this are not committed as pages; their
        strips are returned as singletons at the end.

    Returns
    -------
    pages : list[list[int]]
        Ordered list of pages, each page being an ordered list of strip
        indices.  Unassigned strips appear as singleton lists at the end.

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.array([
    ...     [0.0, 0.9, 0.1],
    ...     [0.1, 0.0, 0.8],
    ...     [0.2, 0.1, 0.0],
    ... ])
    >>> solve_greedy(scores, threshold=0.5, min_chain_length=2)
    [[0, 1, 2]]
    """
    n = score_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    logger.info(
        "Greedy solver: n=%d, threshold=%.2f, min_chain_length=%d",
        n, threshold, min_chain_length,
    )

    edges = [(i, j, float(score_matrix[i, j])) for i in range(n) for j in range(n) if i != j]
    edges.sort(key=lambda x: x[2], reverse=True)

    next_node: dict[int, int] = {}
    prev_node: dict[int, int] = {}
    
    for u, v, score in edges:
        if score < threshold:
            break
        if u in next_node or v in prev_node:
            continue
        # Prevent forming cycles
        curr = v
        forms_cycle = False
        while curr in next_node:
            curr = next_node[curr]
            if curr == u:
                forms_cycle = True
                break
        if not forms_cycle:
            next_node[u] = v
            prev_node[v] = u

    # Build resulting chains from root nodes
    visited = set()
    pages: list[list[int]] = []
    starts = [i for i in range(n) if i not in prev_node]
    
    for start in starts:
        if start in visited:
            continue
        chain = []
        curr = start
        while curr is not None:
            chain.append(curr)
            visited.add(curr)
            curr = next_node.get(curr)
            
        if len(chain) >= min_chain_length:
            pages.append(chain)
        else:
            for idx in chain:
                pages.append([idx])

    for i in range(n):
        if i not in visited:
            pages.append([i])

    logger.info(
        "Greedy solver finished: %d pages (%d multi-strip, %d singletons)",
        len(pages),
        sum(1 for p in pages if len(p) > 1),
        sum(1 for p in pages if len(p) == 1),
    )
    return pages


def solve_greedy_with_clusters(
    score_matrix: np.ndarray,
    clusters: dict[int, list[int]],
    threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Run greedy ordering independently within each pre-assigned cluster.

    This is useful when page clustering has already been performed and
    we only need to determine strip *ordering* within each page.

    Parameters
    ----------
    score_matrix : np.ndarray
        Full n×n score matrix.
    clusters : dict[int, list[int]]
        Mapping from cluster label → list of strip indices.
    threshold : float
        Extension threshold for the greedy solver.

    Returns
    -------
    ordered_clusters : dict[int, list[int]]
        Same cluster labels, but strip indices are now ordered.
    """
    ordered: dict[int, list[int]] = {}

    for label, indices in clusters.items():
        if len(indices) <= 1:
            ordered[label] = list(indices)
            continue

        # Build a sub-matrix for this cluster
        idx_map = {original: local for local, original in enumerate(indices)}
        sub_n = len(indices)
        sub_matrix = np.zeros((sub_n, sub_n), dtype=score_matrix.dtype)
        for li, oi in enumerate(indices):
            for lj, oj in enumerate(indices):
                sub_matrix[li, lj] = score_matrix[oi, oj]

        # Solve within sub-matrix
        sub_pages = solve_greedy(sub_matrix, threshold=threshold, min_chain_length=1)

        # Flatten and map back to original indices
        flat_order = []
        for page in sub_pages:
            for local_idx in page:
                flat_order.append(indices[local_idx])
        ordered[label] = flat_order

    return ordered


def solve_kruskal_greedy(
    score_matrix: np.ndarray,
    threshold: float = 0.3,
    min_chain_length: int = 2,
) -> list[list[int]]:
    """Kruskal-style global edge-selection solver for strip ordering.

    Instead of building chains from a single start node, this solver
    globally sorts ALL directed edges by confidence and greedily accepts
    the best edge that doesn't violate path constraints:
      * Each node has at most one outgoing edge (right neighbor).
      * Each node has at most one incoming edge (left neighbor).
      * No cycles are formed until all nodes are connected.

    This is analogous to Kruskal's MST algorithm applied to directed
    Hamiltonian path construction.

    Parameters
    ----------
    score_matrix : np.ndarray
        n×n asymmetric score matrix.  ``score_matrix[i, j]`` is the
        probability that strip *j* is immediately right of strip *i*.
    threshold : float
        Minimum edge confidence to accept (default: 0.3, lower than
        the chain-based greedy since we're selecting globally).
    min_chain_length : int
        Chains shorter than this are not committed as pages; their
        strips are returned as singletons at the end.

    Returns
    -------
    pages : list[list[int]]
        Ordered list of pages, each page being an ordered list of strip
        indices.  Unassigned strips appear as singleton lists at the end.

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.array([
    ...     [0.0, 0.9, 0.1],
    ...     [0.1, 0.0, 0.8],
    ...     [0.2, 0.1, 0.0],
    ... ])
    >>> solve_kruskal_greedy(scores, threshold=0.5)
    [[0, 1, 2]]
    """
    n = score_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    logger.info(
        "Kruskal greedy solver: n=%d, threshold=%.2f",
        n, threshold,
    )

    # Collect all directed edges (i -> j) with score above threshold
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and score_matrix[i, j] >= threshold:
                edges.append((i, j, float(score_matrix[i, j])))

    # Sort by descending score (best edges first)
    edges.sort(key=lambda x: x[2], reverse=True)

    # Track which nodes have outgoing/incoming edges assigned
    has_right = {}   # node -> its right neighbor
    has_left = {}    # node -> its left neighbor

    def find_chain_end(start: int) -> int:
        """Follow right-links from start to find the chain's rightmost node."""
        curr = start
        while curr in has_right:
            curr = has_right[curr]
        return curr

    def find_chain_start(end: int) -> int:
        """Follow left-links from end to find the chain's leftmost node."""
        curr = end
        while curr in has_left:
            curr = has_left[curr]
        return curr

    def would_form_cycle(u: int, v: int) -> bool:
        """Check if adding edge u->v would create a cycle."""
        # A cycle forms if v is already an ancestor of u in the chain
        curr = find_chain_start(u)
        return curr == v

    accepted = 0
    for u, v, score in edges:
        # Skip if u already has a right neighbor
        if u in has_right:
            continue
        # Skip if v already has a left neighbor
        if v in has_left:
            continue
        # Skip if it would form a cycle
        if would_form_cycle(u, v):
            continue

        has_right[u] = v
        has_left[v] = u
        accepted += 1

        # Stop early if we've connected all possible edges (n-1 for a single path)
        if accepted >= n - 1:
            break

    # Build chains by finding all chain starts (nodes with no left neighbor)
    visited = set()
    pages = []
    singletons = []

    for i in range(n):
        if i in visited:
            continue
        if i in has_left:
            continue  # Not a chain start

        # Build chain from this start
        chain = []
        curr = i
        while curr is not None:
            chain.append(curr)
            visited.add(curr)
            curr = has_right.get(curr)

        if len(chain) >= min_chain_length:
            pages.append(chain)
        else:
            singletons.extend(chain)

    # Any remaining unvisited nodes (shouldn't happen, but just in case)
    for i in range(n):
        if i not in visited:
            singletons.append(i)

    # Singletons become individual pages
    for idx in sorted(singletons):
        pages.append([idx])

    logger.info(
        "Kruskal greedy solver finished: %d pages (%d multi-strip, %d singletons), "
        "%d edges accepted",
        len(pages),
        sum(1 for p in pages if len(p) > 1),
        sum(1 for p in pages if len(p) == 1),
        accepted,
    )
    return pages
