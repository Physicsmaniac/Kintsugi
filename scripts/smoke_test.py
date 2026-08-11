#!/usr/bin/env python3
"""Smoke tests for all core modules.

Run with: python scripts/smoke_test.py
Or:       make test
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    errors = []

    # --- 1. Models ---
    try:
        import torch
        from src.models.seam_model import SeamResNet
        m = SeamResNet(pretrained=False)
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
        print("✅ SeamResNet")
    except Exception as e:
        errors.append(f"SeamResNet: {e}")
        print(f"❌ SeamResNet: {e}")

    try:
        from src.models.page_embedder import PageEmbeddingNet, SupConLoss
        e = PageEmbeddingNet(128, pretrained=False)
        emb = e(torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 128), f"Expected (2,128), got {emb.shape}"
        loss = SupConLoss(0.07)(emb, torch.tensor([0, 0]))
        assert loss.requires_grad
        print(f"✅ PageEmbeddingNet + SupConLoss (loss={loss.item():.4f})")
    except Exception as e:
        errors.append(f"PageEmbeddingNet: {e}")
        print(f"❌ PageEmbeddingNet: {e}")

    # --- 2. Preprocessing ---
    try:
        from PIL import Image
        from src.data.preprocessing import (
            resize_to_training_width, create_seam_pair, get_crops,
            preprocess_pair, preprocess_single_strip, to_grayscale_rgb,
        )
        img = Image.new("RGB", (100, 300), (128, 128, 128))
        assert resize_to_training_width(img, 32).size == (32, 96)
        combined = create_seam_pair(img, img)
        crops = get_crops(combined)
        assert len(crops) == 3
        tensor = preprocess_single_strip(img)
        assert tensor.shape == (3, 224, 224)
        gray = to_grayscale_rgb(img)
        assert gray.mode == "RGB"
        print("✅ Preprocessing")
    except Exception as e:
        errors.append(f"Preprocessing: {e}")
        print(f"❌ Preprocessing: {e}")

    # --- 3. Greedy solver ---
    try:
        import numpy as np
        from src.solver.greedy import solve_greedy
        sm = np.array([[-1, 0.9, 0.1], [0.1, -1, 0.8], [0.3, 0.1, -1]])
        pages = solve_greedy(sm, threshold=0.5)
        assert len(pages) > 0, "No pages found"
        print(f"✅ Greedy solver: {pages}")
    except Exception as e:
        errors.append(f"Greedy solver: {e}")
        print(f"❌ Greedy solver: {e}")

    # --- 4. ATSP solver ---
    try:
        from src.solver.atsp import solve_atsp
        ordering = solve_atsp(sm, temperature=0.5)
        print(f"✅ ATSP solver: {ordering}")
    except ImportError:
        print("⚠️  ATSP solver: python-tsp not installed (optional)")
    except Exception as e:
        errors.append(f"ATSP solver: {e}")
        print(f"❌ ATSP solver: {e}")

    # --- 5. Scoring ---
    try:
        from src.solver.scoring import compute_score_matrix
        imgs = [Image.new("RGB", (50, 200), (i * 40, i * 40, i * 40)) for i in range(3)]
        s, l = compute_score_matrix(m, torch.device("cpu"), imgs)
        assert s.shape == (3, 3), f"Expected (3,3), got {s.shape}"
        print(f"✅ Score matrix: {s.shape}")
    except Exception as e:
        errors.append(f"Scoring: {e}")
        print(f"❌ Scoring: {e}")

    # --- 6. Clustering ---
    try:
        from src.solver.clustering import cluster_strips_by_page
        embs = np.random.randn(20, 128).astype(np.float32)
        embs[:10] += 5  # Two clear clusters
        clusters = cluster_strips_by_page(embs, min_cluster_size=3)
        print(f"✅ Clustering: {len(clusters)} clusters from 20 strips")
    except ImportError:
        print("⚠️  Clustering: hdbscan/umap not installed (optional)")
    except Exception as e:
        errors.append(f"Clustering: {e}")
        print(f"❌ Clustering: {e}")

    # --- 7. Evaluation metrics ---
    try:
        from src.evaluation.metrics import (
            compute_pairwise_accuracy,
            compute_kendall_tau,
            compute_mean_displacement,
        )
        assert compute_pairwise_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
        assert compute_pairwise_accuracy([0, 1, 2], [2, 1, 0]) == 0.0
        kt = compute_kendall_tau([0, 1, 2, 3], [0, 1, 2, 3])
        assert kt == 1.0
        md = compute_mean_displacement([0, 1, 2], [2, 1, 0])
        assert md > 0
        print(f"✅ Metrics (PA=1.0, τ={kt}, disp={md:.1f})")
    except Exception as e:
        errors.append(f"Metrics: {e}")
        print(f"❌ Metrics: {e}")

    # --- 8. Shredder (import only) ---
    try:
        from src.data.shredder import shred_pdf, shred_image
        print("✅ Shredder (import OK)")
    except Exception as e:
        errors.append(f"Shredder: {e}")
        print(f"❌ Shredder: {e}")

    # --- Summary ---
    print()
    if errors:
        print(f"💥 {len(errors)} test(s) failed:")
        for err in errors:
            print(f"   ❌ {err}")
        return 1
    else:
        print("🎉 All smoke tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
