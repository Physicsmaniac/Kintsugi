"""Page clustering for multi-page shredded document reconstruction.

Pipeline:
    1. **Dimensionality reduction** — PCA(50) → UMAP(10)
    2. **Clustering** — HDBSCAN on the reduced embeddings
    3. **Noise reassignment** — hybrid embedding-distance + seam-score metric
    4. **Block detection** — spectral eigengap analysis to split merged clusters

All heavy dependencies (UMAP, HDBSCAN, scikit-learn) are imported lazily so
the module remains importable even when they are missing.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------

_SKLEARN_AVAILABLE: bool = False
try:
    from sklearn.decomposition import PCA  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    from sklearn.metrics import pairwise_distances  # type: ignore[import-untyped]

    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

_UMAP_AVAILABLE: bool = False
try:
    import umap  # type: ignore[import-untyped]

    _UMAP_AVAILABLE = True
except ImportError:
    pass

_HDBSCAN_AVAILABLE: bool = False
try:
    import hdbscan  # type: ignore[import-untyped]

    _HDBSCAN_AVAILABLE = True
except ImportError:
    pass


def _check_dependencies(need_umap: bool = True) -> None:
    """Raise informative errors for missing optional packages."""
    missing: list[str] = []
    if not _SKLEARN_AVAILABLE:
        missing.append("scikit-learn")
    if need_umap and not _UMAP_AVAILABLE:
        missing.append("umap-learn")
    if not _HDBSCAN_AVAILABLE:
        missing.append("hdbscan")
    if missing:
        raise ImportError(
            f"The following packages are required for clustering but not "
            f"installed: {', '.join(missing)}.  "
            f"Install them with:  pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers
# ---------------------------------------------------------------------------

def _reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Standardise and PCA-reduce embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        (n_samples, n_features) raw embeddings.
    n_components : int
        Target PCA dimensionality; capped at min(n_samples, n_features).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    reduced : np.ndarray
        (n_samples, min(n_components, …)) PCA-projected embeddings.
    """
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA: %d → %d dimensions (%.1f%% variance explained)",
        embeddings.shape[1], n_components, 100.0 * explained,
    )
    return reduced


def _reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """UMAP reduction (expects PCA-pre-reduced input).

    Parameters
    ----------
    embeddings : np.ndarray
        (n_samples, n_features) — typically PCA-reduced.
    n_components : int
        UMAP output dimensionality.
    n_neighbors : int
        UMAP locality parameter.
    min_dist : float
        Minimum distance in the embedding.
    metric : str
        Distance metric.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    reduced : np.ndarray
        (n_samples, n_components) UMAP embedding.
    """
    reducer = umap.UMAP(
        n_components=min(n_components, embeddings.shape[0] - 2),
        n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)
    logger.info(
        "UMAP: %d → %d dimensions",
        embeddings.shape[1], reduced.shape[1],
    )
    return reduced


# ---------------------------------------------------------------------------
# Main clustering entry-point
# ---------------------------------------------------------------------------

def cluster_strips_by_page(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    use_umap: bool = True,
    umap_n_components: int = 10,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """Cluster strip embeddings into pages using HDBSCAN.

    Parameters
    ----------
    embeddings : np.ndarray
        (n_strips, feature_dim) embedding vectors (e.g. from the
        classifier's penultimate layer).
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    use_umap : bool
        If ``True``, apply PCA(50) → UMAP(``umap_n_components``) before
        clustering.  Otherwise only PCA(50) is used.
    umap_n_components : int
        Number of UMAP output dimensions.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    clusters : dict[int, list[int]]
        Mapping from cluster label → list of strip indices.
        Noise strips are assigned label ``-1``.
    """
    _check_dependencies(need_umap=use_umap)

    n = embeddings.shape[0]
    logger.info("Clustering %d strips (min_cluster_size=%d, use_umap=%s)",
                n, min_cluster_size, use_umap)

    if n < min_cluster_size:
        logger.warning(
            "Fewer strips (%d) than min_cluster_size (%d); "
            "returning all strips in a single cluster.",
            n, min_cluster_size,
        )
        return {0: list(range(n))}

    # --- Dimensionality reduction ---
    if use_umap:
        reduced = _reduce_pca(embeddings, n_components=50, random_state=random_state)
        if _UMAP_AVAILABLE and reduced.shape[0] > umap_n_components + 2:
            reduced = _reduce_umap(
                reduced,
                n_components=umap_n_components,
                random_state=random_state,
            )
    else:
        # Use raw L2-normalized embeddings directly (StandardScaler destroys hypersphere topology)
        reduced = embeddings

    # --- HDBSCAN ---
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    # Build cluster dict
    clusters: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(idx)

    n_clusters = sum(1 for k in clusters if k >= 0)
    n_noise = len(clusters.get(-1, []))
    logger.info(
        "HDBSCAN found %d clusters and %d noise strips.", n_clusters, n_noise,
    )

    return clusters


# ---------------------------------------------------------------------------
# Noise reassignment
# ---------------------------------------------------------------------------

def reassign_noise_strips(
    noise_indices: list[int],
    clusters: dict[int, list[int]],
    embeddings: np.ndarray,
    score_matrix: np.ndarray | None = None,
    embedding_weight: float = 0.4,
    seam_weight: float = 0.6,
) -> dict[int, int]:
    """Reassign noise strips to the closest existing cluster.

    Uses a hybrid metric combining:
        * **Embedding distance** — cosine distance from the noise strip to
          each cluster centroid (in the original embedding space).
        * **Seam score** — average pairwise adjacency score between the noise
          strip and all members of the cluster.

    Parameters
    ----------
    noise_indices : list[int]
        Strip indices labelled as noise (``-1``).
    clusters : dict[int, list[int]]
        Current cluster assignments (excluding noise).
    embeddings : np.ndarray
        (n_strips, dim) embedding matrix.
    score_matrix : np.ndarray | None
        n×n adjacency score matrix.  If ``None``, only embedding distance
        is used (``embedding_weight`` is set to 1.0).
    embedding_weight : float
        Weight for embedding-distance component.
    seam_weight : float
        Weight for seam-score component.

    Returns
    -------
    assignments : dict[int, int]
        Mapping ``{noise_strip_index: assigned_cluster_label}``.
    """
    _check_dependencies(need_umap=False)

    if not noise_indices:
        return {}

    # Filter to real clusters only (label ≥ 0)
    real_clusters = {k: v for k, v in clusters.items() if k >= 0}
    if not real_clusters:
        logger.warning("No real clusters to reassign noise strips to.")
        return {}

    if score_matrix is None:
        embedding_weight = 1.0
        seam_weight = 0.0
        logger.info("No score matrix provided; using embedding distance only.")

    # Normalise weights
    total = embedding_weight + seam_weight
    if total <= 0:
        embedding_weight, seam_weight = 0.5, 0.5
        total = 1.0
    embedding_weight /= total
    seam_weight /= total

    # Precompute cluster centroids
    centroids: dict[int, np.ndarray] = {}
    for label, members in real_clusters.items():
        centroids[label] = embeddings[members].mean(axis=0)

    assignments: dict[int, int] = {}

    for ni in noise_indices:
        best_label = -1
        best_combined = float("inf")

        emb_i = embeddings[ni]

        for label, members in real_clusters.items():
            # Embedding distance (cosine)
            centroid = centroids[label]
            cos_dist = 1.0 - float(
                np.dot(emb_i, centroid)
                / (np.linalg.norm(emb_i) * np.linalg.norm(centroid) + 1e-12)
            )

            # Seam score (average with cluster members → lower is better)
            if score_matrix is not None and seam_weight > 0:
                scores_from = [float(score_matrix[ni, m]) for m in members]
                scores_to = [float(score_matrix[m, ni]) for m in members]
                avg_seam = (sum(scores_from) + sum(scores_to)) / (2 * len(members))
                seam_cost = 1.0 - avg_seam
            else:
                seam_cost = 0.0

            combined = embedding_weight * cos_dist + seam_weight * seam_cost
            if combined < best_combined:
                best_combined = combined
                best_label = label

        assignments[ni] = best_label

    logger.info(
        "Reassigned %d noise strips across %d clusters.",
        len(assignments), len(set(assignments.values())),
    )
    return assignments


# ---------------------------------------------------------------------------
# Spectral block detection (split merged clusters)
# ---------------------------------------------------------------------------

def detect_and_split_merged_clusters(
    cluster_indices: list[int],
    score_matrix: np.ndarray,
    eigengap_threshold: float = 0.3,
) -> list[list[int]] | None:
    """Detect whether a cluster should be split using spectral gap analysis.

    Builds a within-cluster affinity matrix from the score matrix, computes
    the graph Laplacian, and looks for a significant eigengap that indicates
    the cluster actually contains two or more blocks.

    Parameters
    ----------
    cluster_indices : list[int]
        Strip indices belonging to this cluster.
    score_matrix : np.ndarray
        Full n×n adjacency score matrix.
    eigengap_threshold : float
        Minimum gap between consecutive sorted eigenvalues (normalised by
        the largest eigenvalue) to declare a split.

    Returns
    -------
    sub_clusters : list[list[int]] | None
        If a split is detected, returns a list of sub-clusters (each a list
        of original strip indices).  ``None`` if no split is warranted.
    """
    _check_dependencies(need_umap=False)

    k = len(cluster_indices)
    if k < 4:
        return None

    # Build symmetric affinity from the asymmetric score matrix
    affinity = np.zeros((k, k), dtype=np.float64)
    for li, oi in enumerate(cluster_indices):
        for lj, oj in enumerate(cluster_indices):
            if li != lj:
                affinity[li, lj] = (
                    score_matrix[oi, oj] + score_matrix[oj, oi]
                ) / 2.0

    # Graph Laplacian: L = D − A
    degree = affinity.sum(axis=1)
    laplacian = np.diag(degree) - affinity

    # Compute eigenvalues of the normalised Laplacian
    # Using D^{-1/2} L D^{-1/2} for stability
    d_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 1e-12
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = D_inv_sqrt @ laplacian @ D_inv_sqrt

    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_norm)))

    # Find the largest eigengap (skip the first eigenvalue which is ≈ 0)
    max_gap = 0.0
    split_k = 1
    scale = max(eigenvalues[-1], 1e-12)

    for i in range(1, min(len(eigenvalues) - 1, k // 2 + 1)):
        gap = (eigenvalues[i + 1] - eigenvalues[i]) / scale
        if gap > max_gap:
            max_gap = gap
            split_k = i + 1

    logger.debug(
        "Eigengap analysis (cluster size=%d): max_gap=%.4f at k=%d (threshold=%.2f)",
        k, max_gap, split_k, eigengap_threshold,
    )

    if max_gap < eigengap_threshold or split_k <= 1:
        return None

    # Spectral clustering with split_k clusters
    from sklearn.cluster import SpectralClustering  # type: ignore[import-untyped]

    sc = SpectralClustering(
        n_clusters=split_k,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans",
    )
    sub_labels = sc.fit_predict(affinity)

    sub_clusters: list[list[int]] = []
    for c in range(split_k):
        members = [cluster_indices[i] for i in range(k) if sub_labels[i] == c]
        if members:
            sub_clusters.append(members)

    if len(sub_clusters) <= 1:
        return None

    logger.info(
        "Split cluster of %d strips into %d sub-clusters (eigengap=%.3f): %s",
        k, len(sub_clusters), max_gap,
        [len(sc) for sc in sub_clusters],
    )
    return sub_clusters


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def cluster_and_refine(
    embeddings: np.ndarray,
    score_matrix: np.ndarray | None = None,
    min_cluster_size: int = 3,
    use_umap: bool = True,
    umap_n_components: int = 10,
    eigengap_threshold: float = 0.3,
    embedding_weight: float = 0.4,
    seam_weight: float = 0.6,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """Full clustering pipeline: cluster → split → reassign noise.

    Parameters
    ----------
    embeddings : np.ndarray
        (n, d) strip embedding matrix.
    score_matrix : np.ndarray | None
        n×n adjacency score matrix (optional, improves noise reassignment
        and split detection).
    min_cluster_size : int
        HDBSCAN minimum cluster size.
    use_umap : bool
        Whether to apply UMAP after PCA.
    umap_n_components : int
        UMAP target dimensionality.
    eigengap_threshold : float
        Threshold for spectral split detection.
    embedding_weight : float
        Hybrid reassignment weight for embedding distance.
    seam_weight : float
        Hybrid reassignment weight for seam score.
    random_state : int
        Random seed.

    Returns
    -------
    final_clusters : dict[int, list[int]]
        Refined cluster assignments, noise-free.
    """
    # Step 1: initial clustering
    clusters = cluster_strips_by_page(
        embeddings,
        min_cluster_size=min_cluster_size,
        use_umap=use_umap,
        umap_n_components=umap_n_components,
        random_state=random_state,
    )

    # Step 2: detect and split merged clusters
    if score_matrix is not None:
        new_label = max(clusters.keys()) + 1
        to_add: dict[int, list[int]] = {}
        to_remove: list[int] = []

        for label, members in list(clusters.items()):
            if label < 0:
                continue
            splits = detect_and_split_merged_clusters(
                members, score_matrix, eigengap_threshold=eigengap_threshold,
            )
            if splits is not None and len(splits) > 1:
                to_remove.append(label)
                for sub in splits:
                    to_add[new_label] = sub
                    new_label += 1

        for lbl in to_remove:
            del clusters[lbl]
        clusters.update(to_add)

    # Step 3: reassign noise
    noise = clusters.pop(-1, [])
    if noise:
        assignments = reassign_noise_strips(
            noise_indices=noise,
            clusters=clusters,
            embeddings=embeddings,
            score_matrix=score_matrix,
            embedding_weight=embedding_weight,
            seam_weight=seam_weight,
        )
        for strip_idx, target_label in assignments.items():
            clusters.setdefault(target_label, []).append(strip_idx)

    # Renumber clusters contiguously from 0
    final: dict[int, list[int]] = {}
    for new_id, (_, members) in enumerate(sorted(clusters.items())):
        final[new_id] = sorted(members)

    logger.info(
        "Clustering pipeline complete: %d final clusters, sizes=%s",
        len(final), [len(v) for v in final.values()],
    )
    return final


def build_joint_affinity(
    embeddings: np.ndarray,
    score_matrix: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Build a joint affinity matrix combining embedding similarity and seam scores."""
    # Cosine similarity from embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = embeddings / norms
    cos_sim = normalized @ normalized.T
    
    # Symmetrize seam scores
    seam_sym = (score_matrix + score_matrix.T) / 2.0
    
    # Combine
    affinity = alpha * cos_sim + (1 - alpha) * seam_sym
    np.fill_diagonal(affinity, 0.0)
    return affinity

def cluster_spectral(
    affinity: np.ndarray,
    num_clusters: int,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """Spectral clustering with known number of clusters.
    
    Much more reliable than HDBSCAN for small sample sizes (20-100 strips).
    """
    from sklearn.cluster import SpectralClustering
    
    n = affinity.shape[0]
    if n <= num_clusters:
        return {i: [i] for i in range(n)}
    
    # Ensure affinity is non-negative for spectral clustering
    affinity_nn = affinity - affinity.min()
    np.fill_diagonal(affinity_nn, 0.0)
    
    sc = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=random_state,
        assign_labels='kmeans',
        n_init=10,
    )
    labels = sc.fit_predict(affinity_nn)
    
    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(idx)
    
    logger.info("Spectral clustering: %d clusters, sizes: %s",
                len(clusters), [len(v) for v in clusters.values()])
    return clusters

def cluster_and_refine_joint(
    embeddings: np.ndarray,
    score_matrix: np.ndarray,
    num_pages: int,
    alpha: float = 0.5,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """Full pipeline using joint affinity + spectral clustering.
    
    This is the recommended approach when num_pages is known.
    """
    affinity = build_joint_affinity(embeddings, score_matrix, alpha=alpha)
    clusters = cluster_spectral(affinity, num_clusters=num_pages, random_state=random_state)
    return clusters
