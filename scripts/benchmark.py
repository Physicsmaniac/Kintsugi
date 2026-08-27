from __future__ import annotations

import argparse
import numpy as np
import csv
import logging
import os
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.models.seam_model import load_seam_model
from src.models.page_embedder import load_page_embedder
from src.solver.scoring import compute_score_matrix
from src.solver.greedy import solve_greedy, solve_kruskal_greedy, solve_greedy_with_clusters
from src.solver.atsp import solve_atsp
from src.solver.clustering import (
    cluster_and_refine,
    cluster_and_refine_joint,
    cluster_spectral,
    build_joint_affinity,
)
from src.evaluation.metrics import evaluate_reconstruction
from src.data.shredder import shred_pdf
from src.data.preprocessing import to_grayscale_rgb, preprocess_single_strip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_experiment(
    pdf_path: str,
    output_dir: str,
    num_pages: int,
    num_strips: int,
    strategy: str,
    seam_model: torch.nn.Module,
    page_model: torch.nn.Module | None,
    device: torch.device,
    greedy_threshold: float = 0.5,
) -> Any:
    """Run a single benchmark experiment."""
    logger.info(
        f"Running experiment: pages={num_pages}, strips={num_strips}, strategy={strategy}"
    )

    shred_dir = Path(output_dir) / f"shredded_{num_pages}p_{num_strips}s"
    shred_dir.mkdir(parents=True, exist_ok=True)

    # Shred PDF -> returns dict: { "strip_0000.jpg": {"page": 0, "index": 3}, ... }
    strips_info = shred_pdf(
        pdf_path=pdf_path,
        output_dir=str(shred_dir),
        num_strips=num_strips,
        max_pages=num_pages,
        shuffle=True,
        seed=42,
    )

    if not strips_info:
        logger.warning("No strips generated. Returning empty results.")
        from src.evaluation.metrics import ReconstructionMetrics

        return ReconstructionMetrics()

    # Load and preprocess strips
    images = []
    true_page_labels_dict = {}
    true_pages_map = defaultdict(list)

    for strip_idx, (fname, meta) in enumerate(strips_info.items()):
        img_path = shred_dir / fname
        page_idx = meta["page"]
        orig_idx = meta["index"]

        img = Image.open(img_path).convert("RGB")
        images.append(img)
        true_page_labels_dict[strip_idx] = page_idx
        true_pages_map[page_idx].append((orig_idx, strip_idx))

    # Build true_pages dict mapping page_idx -> list of strip_indices in correct left-to-right order
    true_pages = {}
    for page_idx, strip_list in true_pages_map.items():
        strip_list.sort(key=lambda x: x[0])
        true_pages[page_idx] = [s_idx for _, s_idx in strip_list]

    true_label_array = np.array([true_page_labels_dict[i] for i in range(len(images))])

    # Compute score matrix efficiently using the batched version
    from src.solver.scoring import compute_score_matrix_batched
    score_matrix, _ = compute_score_matrix_batched(seam_model, device, images, batch_size=64, num_workers=4)

    pred_pages = []
    pred_page_labels = np.full(len(images), -1, dtype=int)
    is_clustered = "hdbscan" in strategy or "spectral" in strategy

    if is_clustered:
        if page_model is None:
            raise ValueError(f"Strategy {strategy} requires page_model.")

        embeddings = []
        page_model.eval()
        with torch.no_grad():
            for img in images:
                tensor = preprocess_single_strip(img).unsqueeze(0).to(device)
                emb = page_model(tensor)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                embeddings.append(emb.squeeze(0).cpu().numpy())

        embeddings_arr = np.array(embeddings)
        
        if "spectral" in strategy:
            clusters = cluster_and_refine_joint(
                embeddings=embeddings_arr,
                score_matrix=score_matrix,
                num_pages=num_pages,
                alpha=0.5,
            )
        else:
            clusters = cluster_and_refine(
                embeddings=embeddings_arr,
                score_matrix=score_matrix,
                min_cluster_size=max(2, num_strips // 2),
                use_umap=True,
            )

        for cluster_id, strip_indices in clusters.items():
            if len(strip_indices) == 0:
                continue

            for idx in strip_indices:
                pred_page_labels[idx] = cluster_id

            sub_matrix = score_matrix[np.ix_(strip_indices, strip_indices)]

            if "kruskal" in strategy:
                from src.solver.greedy import solve_kruskal_greedy
                chains = solve_kruskal_greedy(sub_matrix, threshold=greedy_threshold)
                for chain in chains:
                    pred_pages.append([strip_indices[i] for i in chain])
            elif "greedy" in strategy:
                chains = solve_greedy(sub_matrix, threshold=greedy_threshold)
                for chain in chains:
                    pred_pages.append([strip_indices[i] for i in chain])
            elif "atsp" in strategy:
                order = solve_atsp(sub_matrix, temperature=1.0, use_log_cost=True)
                pred_pages.append([strip_indices[i] for i in order])
            else:
                raise ValueError(f"Unknown solver in strategy {strategy}")
    else:
        # Global solver without clustering
        if "kruskal" in strategy:
            from src.solver.greedy import solve_kruskal_greedy
            chains = solve_kruskal_greedy(score_matrix, threshold=greedy_threshold)
            pred_pages = chains
        elif "greedy" in strategy:
            chains = solve_greedy(score_matrix, threshold=greedy_threshold)
            pred_pages = chains
        elif "atsp" in strategy:
            global_order = solve_atsp(score_matrix, temperature=1.0, use_log_cost=True)
            pred_pages = [global_order]
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    metrics = evaluate_reconstruction(
        pred_pages=pred_pages,
        true_pages=true_pages,
        true_page_labels=true_page_labels_dict,
        pred_page_labels=pred_page_labels if is_clustered else None,
        true_label_array=true_label_array if is_clustered else None,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark thesis experiments")
    parser.add_argument("--pdf", type=str, required=True, help="Path to test PDF")
    parser.add_argument(
        "--seam-model", type=str, required=True, help="Path to seam model weights"
    )
    parser.add_argument(
        "--page-model", type=str, default=None, help="Path to page model weights"
    )
    parser.add_argument(
        "--num-pages", type=int, nargs="+", default=[2, 5], help="List of page counts"
    )
    parser.add_argument(
        "--num-strips",
        type=int,
        nargs="+",
        default=[10, 20],
        help="List of strip counts",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["kruskal", "atsp", "spectral+kruskal", "spectral+atsp", "spectral+greedy", "hdbscan+greedy", "hdbscan+atsp"],
        help="Strategies to run",
    )
    parser.add_argument(
        "--greedy-thresholds",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of greedy thresholds to evaluate",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    seam_model, _ = load_seam_model(args.seam_model, device)

    page_model = None
    if args.page_model and any(c in s for s in args.strategies for c in ["hdbscan", "spectral"]):
        page_model, _ = load_page_embedder(args.page_model, device)

    results = []
    
    csv_path = Path(args.output_dir) / "benchmark_results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["num_pages", "num_strips", "strategy", "greedy_threshold", "ari", "pairwise_accuracy"]
        )

        for p, s, strat in itertools.product(
            args.num_pages, args.num_strips, args.strategies
        ):
            thresholds = args.greedy_thresholds if any(k in strat for k in ["greedy", "kruskal"]) else [0.0]
            
            for threshold in thresholds:
                metrics = run_experiment(
                    args.pdf, args.output_dir, p, s, strat, seam_model, page_model, device, greedy_threshold=threshold
                )

                ari = getattr(metrics.clustering, "adjusted_rand_index", 0.0)
                p_acc = getattr(metrics.ordering, "pairwise_accuracy", 0.0)

                writer.writerow([p, s, strat, threshold, ari, p_acc])
                results.append(
                    {
                        "num_pages": p,
                        "num_strips": s,
                        "strategy": strat,
                        "greedy_threshold": threshold,
                        "ari": ari,
                        "pairwise_accuracy": p_acc,
                    }
                )

                logger.info(
                    f"Results for {p}p {s}s {strat} (th={threshold}): ARI={ari:.4f}, P_Acc={p_acc:.4f}"
                )

    # Generate plots
    # Bar chart for ARI by strategy
    strategies = args.strategies
    avg_ari = {strat: 0.0 for strat in strategies}
    avg_p_acc = {strat: 0.0 for strat in strategies}
    counts = {strat: 0 for strat in strategies}

    for r in results:
        strat = r["strategy"]
        avg_ari[strat] += r["ari"]
        avg_p_acc[strat] += r["pairwise_accuracy"]
        counts[strat] += 1

    for strat in strategies:
        if counts[strat] > 0:
            avg_ari[strat] /= counts[strat]
            avg_p_acc[strat] /= counts[strat]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(strategies, [avg_ari[s] for s in strategies], color="skyblue")
    ax1.set_title("Average ARI by Strategy")
    ax1.set_ylabel("ARI")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(strategies, [avg_p_acc[s] for s in strategies], color="lightgreen")
    ax2.set_title("Average Pairwise Accuracy by Strategy")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "strategy_comparison.png")
    logger.info(f"Plots saved to {args.output_dir}/strategy_comparison.png")


if __name__ == "__main__":
    main()
