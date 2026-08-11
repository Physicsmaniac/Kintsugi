from __future__ import annotations

import argparse
import csv
import logging
import os
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.models.seam_model import load_seam_model
from src.models.page_embedder import load_page_embedder
from src.solver.scoring import compute_score_matrix
from src.solver.greedy import solve_greedy
from src.solver.atsp import solve_atsp
from src.solver.clustering import cluster_strips_by_page, detect_and_split_merged_clusters
from src.evaluation.metrics import evaluate_reconstruction
from src.data.shredder import shred_pdf
from src.data.preprocessing import to_grayscale_rgb, preprocess_single_strip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
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
    device: torch.device
) -> dict[str, Any]:
    """Run a single benchmark experiment."""
    logger.info(f"Running experiment: pages={num_pages}, strips={num_strips}, strategy={strategy}")
    
    shred_dir = Path(output_dir) / f"shredded_{num_pages}p_{num_strips}s"
    shred_dir.mkdir(parents=True, exist_ok=True)
    
    # Shred PDF
    strips_info = shred_pdf(
        pdf_path=pdf_path,
        output_dir=str(shred_dir),
        num_strips=num_strips,
        max_pages=num_pages,
        shuffle=True,
        seed=42
    )
    
    if not strips_info:
        logger.warning("No strips generated. Returning empty results.")
        return {}
        
    # Load and preprocess strips
    images = []
    true_page_labels = []
    for info in strips_info:
        img_path = info["path"]
        page_idx = info["page_idx"]
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        true_page_labels.append(page_idx)
        
    true_pages = defaultdict(list)
    for idx, page_lbl in enumerate(true_page_labels):
        true_pages[page_lbl].append(idx)
    true_pages_list = list(true_pages.values())
        
    # Compute score matrix
    score_matrix = compute_score_matrix(seam_model, device, images)
    
    pred_pages = []
    
    if "hdbscan" in strategy:
        if page_model is None:
            raise ValueError("HDBSCAN strategy requires page_model.")
            
        embeddings = []
        page_model.eval()
        with torch.no_grad():
            for img in images:
                tensor = preprocess_single_strip(img).unsqueeze(0).to(device)
                emb = page_model(tensor)
                embeddings.append(emb.squeeze(0).cpu().numpy())
                
        clusters = cluster_strips_by_page(embeddings, min_cluster_size=num_strips // 2)
        clusters = detect_and_split_merged_clusters(clusters, score_matrix)
        
        for cluster_id, strip_indices in clusters.items():
            if len(strip_indices) == 0:
                continue
                
            # Extract submatrix
            sub_matrix = score_matrix[strip_indices][:, strip_indices]
            
            if "greedy" in strategy:
                order = solve_greedy(sub_matrix, threshold=0.5)
            elif "atsp" in strategy:
                order = solve_atsp(sub_matrix, temperature=1.0)
            else:
                raise ValueError(f"Unknown solver in strategy {strategy}")
                
            pred_pages.append([strip_indices[i] for i in order])
    else:
        # Global solver
        if "greedy" in strategy:
            # Assuming greedy can return multiple lists or we just do it globally
            global_order = solve_greedy(score_matrix, threshold=0.5)
            pred_pages = [global_order]
        elif "atsp" in strategy:
            global_order = solve_atsp(score_matrix, temperature=1.0)
            pred_pages = [global_order]
        else:
            raise ValueError(f"Unknown strategy {strategy}")
            
    metrics = evaluate_reconstruction(pred_pages, true_pages_list, true_page_labels)
    
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark thesis experiments")
    parser.add_argument("--pdf", type=str, required=True, help="Path to test PDF")
    parser.add_argument("--seam-model", type=str, required=True, help="Path to seam model weights")
    parser.add_argument("--page-model", type=str, default=None, help="Path to page model weights")
    parser.add_argument("--num-pages", type=int, nargs="+", default=[2, 5], help="List of page counts")
    parser.add_argument("--num-strips", type=int, nargs="+", default=[10, 20], help="List of strip counts")
    parser.add_argument("--strategies", type=str, nargs="+", default=["greedy", "atsp", "hdbscan+greedy", "hdbscan+atsp"], help="Strategies to run")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    seam_model = load_seam_model(args.seam_model, device)
    
    page_model = None
    if args.page_model and any("hdbscan" in s for s in args.strategies):
        page_model = load_page_embedder(args.page_model, device)
        
    results = []
    
    csv_path = Path(args.output_dir) / "benchmark_results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_pages", "num_strips", "strategy", "ari", "pairwise_accuracy"])
        
        for p, s, strat in itertools.product(args.num_pages, args.num_strips, args.strategies):
            metrics = run_experiment(args.pdf, args.output_dir, p, s, strat, seam_model, page_model, device)
            
            ari = metrics.get("ari", 0.0)
            p_acc = metrics.get("pairwise_accuracy", 0.0)
            
            writer.writerow([p, s, strat, ari, p_acc])
            results.append({
                "num_pages": p,
                "num_strips": s,
                "strategy": strat,
                "ari": ari,
                "pairwise_accuracy": p_acc
            })
            
            logger.info(f"Results for {p}p {s}s {strat}: ARI={ari:.4f}, P_Acc={p_acc:.4f}")
            
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
    
    ax1.bar(strategies, [avg_ari[s] for s in strategies], color='skyblue')
    ax1.set_title("Average ARI by Strategy")
    ax1.set_ylabel("ARI")
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(strategies, [avg_p_acc[s] for s in strategies], color='lightgreen')
    ax2.set_title("Average Pairwise Accuracy by Strategy")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "strategy_comparison.png")
    logger.info(f"Plots saved to {args.output_dir}/strategy_comparison.png")
    
if __name__ == "__main__":
    main()
