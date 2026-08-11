"""Forensic Document Reconstructor — Streamlit Application.

Refactored to use the modular src/ package structure. Adds thesis features:
- Solver strategy selection (Greedy / ATSP)
- Page clustering toggle (HDBSCAN)
- Embedding visualization
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import string
import sys

import cv2
import numpy as np
import plotly.express as px
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.seam_model import SeamResNet, load_seam_model
from src.models.page_embedder import PageEmbeddingNet, load_page_embedder
from src.data.preprocessing import (
    to_grayscale_rgb,
    preprocess_single_strip,
    normalize_transform,
)
from src.solver.scoring import compute_score_matrix
from src.solver.greedy import solve_greedy
from src.solver.atsp import solve_atsp

logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensic Reconstruction AI", layout="wide")

# --- SESSION STATE ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def clear_workspace():
    st.session_state.uploader_key += 1
    for key in ["shredded_data", "current_shred"]:
        st.session_state.pop(key, None)


def send_to_solver():
    """Callback to transfer shredded data to solver and switch tabs."""
    if "current_shred" in st.session_state:
        st.session_state["shredded_data"] = st.session_state["current_shred"][
            "generated_strips"
        ]
        del st.session_state["current_shred"]
        st.session_state.active_tab = "🧩 Solver"


def process_uploads(uploaded_files):
    """Convert uploaded files to PIL and OpenCV images."""
    images_pil = []
    images_cv = []
    for file in uploaded_files:
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        images_cv.append(img_cv)
        images_pil.append(img_pil)
    return images_pil, images_cv


def stitch_page(images_cv, chain):
    """Stitch strip images together into a full page."""
    page_imgs = [images_cv[c] for c in chain]
    target_h = page_imgs[0].shape[0]
    resized_imgs = []
    for img in page_imgs:
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            resized_imgs.append(cv2.resize(img, (new_w, target_h)))
        else:
            resized_imgs.append(img)
    return np.concatenate(resized_imgs, axis=1)


# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background-color: #4CAF50;
        border-color: #4CAF50;
        color: white;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    div:has(span.red-btn-marker) + div button {
        background-color: #FF4B4B !important;
        border-color: #FF4B4B !important;
        color: white !important;
    }
    div:has(span.red-btn-marker) + div button:hover {
        background-color: #FF0000 !important;
        border-color: #FF0000 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- TITLE ---
st.title("Forensic Document Reconstructor")

# --- NAVIGATION ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🧩 Solver"

nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    if st.button(
        "🧩 Solver",
        use_container_width=True,
        type="primary"
        if st.session_state.active_tab == "🧩 Solver"
        else "secondary",
    ):
        st.session_state.active_tab = "🧩 Solver"
        st.rerun()

with nav_col2:
    if st.button(
        "📄 Shredder",
        use_container_width=True,
        type="primary"
        if st.session_state.active_tab == "📄 Shredder"
        else "secondary",
    ):
        st.session_state.active_tab = "📄 Shredder"
        st.rerun()

active_tab = st.session_state.active_tab

# ======================================================================
# SHREDDER TAB
# ======================================================================
if active_tab == "📄 Shredder":
    st.header("Document Shredder")
    st.markdown("Upload a PDF or Image to shred into randomized strips.")

    uploaded_file = st.file_uploader(
        "Upload Document", type=["pdf", "jpg", "png", "jpeg"]
    )

    col1, col2 = st.columns(2)
    with col1:
        is_pdf = uploaded_file is not None and uploaded_file.type == "application/pdf"
        page_num = st.number_input(
            "Page Number (PDF only)", min_value=1, value=1, disabled=not is_pdf
        )
    with col2:
        num_strips = st.number_input("Number of Strips", min_value=2, value=15)

    st.markdown('<span class="red-btn-marker"></span>', unsafe_allow_html=True)
    if st.button("Shred & Randomize", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a file first.")
        else:
            with st.spinner("Shredding..."):
                try:
                    page_img = None
                    if uploaded_file.type == "application/pdf":
                        from pdf2image import convert_from_bytes

                        pdf_bytes = uploaded_file.read()
                        images = convert_from_bytes(
                            pdf_bytes,
                            dpi=300,
                            first_page=page_num,
                            last_page=page_num,
                        )
                        if not images:
                            st.error(f"Page {page_num} not found in PDF.")
                        else:
                            page_img = np.array(images[0])
                    else:
                        image = Image.open(uploaded_file).convert("RGB")
                        page_img = np.array(image)

                    if page_img is not None:
                        if page_img.ndim == 3:
                            page_img = cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR)

                        height, width, _ = page_img.shape
                        strip_width = width // num_strips

                        zip_buffer = io.BytesIO()
                        ground_truth = {}
                        generated_strips = []

                        import zipfile

                        with zipfile.ZipFile(
                            zip_buffer, "w", zipfile.ZIP_DEFLATED
                        ) as zf:
                            for i in range(num_strips):
                                x_start = i * strip_width
                                x_end = (
                                    width
                                    if (i == num_strips - 1)
                                    else (i + 1) * strip_width
                                )
                                strip = page_img[:, x_start:x_end]

                                rand_id = "".join(
                                    random.choices(
                                        string.ascii_lowercase + string.digits, k=8
                                    )
                                )
                                filename = f"page_{page_num}_strip_{rand_id}.jpg"

                                is_success, buffer = cv2.imencode(".jpg", strip)
                                if is_success:
                                    img_bytes = buffer.tobytes()
                                    zf.writestr(filename, img_bytes)
                                    generated_strips.append(
                                        {"name": filename, "bytes": img_bytes}
                                    )
                                    ground_truth[filename] = {
                                        "real_index": i,
                                        "x_start": x_start,
                                        "page": page_num,
                                    }

                            zf.writestr(
                                "ground_truth.json",
                                json.dumps(ground_truth, indent=4),
                            )

                        st.session_state["current_shred"] = {
                            "page_num": page_num,
                            "num_strips": num_strips,
                            "zip_bytes": zip_buffer.getvalue(),
                            "generated_strips": generated_strips,
                        }

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    # Display Results (Persistent)
    if "current_shred" in st.session_state:
        shred_data = st.session_state["current_shred"]
        st.success(
            f"Successfully shredded Page {shred_data['page_num']} "
            f"into {shred_data['num_strips']} strips!"
        )

        with st.expander(
            "🕵️ Verify Shredded Strips (Proof of Randomness)", expanded=False
        ):
            st.caption("These are the actual randomized files contained in the ZIP.")
            strips = shred_data["generated_strips"]
            cols_per_row = 10
            for i in range(0, len(strips), cols_per_row):
                cols = st.columns(cols_per_row)
                batch = strips[i : i + cols_per_row]
                for j, strip in enumerate(batch):
                    with cols[j]:
                        st.image(strip["bytes"], use_container_width=True)
                        st.markdown(
                            f"<p style='text-align: center; color: #00E5FF; "
                            f"font-size: 9px; font-weight: bold; line-height: 1.1;'>"
                            f"{strip['name']}</p>",
                            unsafe_allow_html=True,
                        )

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Shredded Strips (ZIP)",
                data=shred_data["zip_bytes"],
                file_name=f"shredded_page_{shred_data['page_num']}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
        with col2:
            st.button(
                "Send to Solver ➡️",
                type="primary",
                use_container_width=True,
                on_click=send_to_solver,
            )

# ======================================================================
# SOLVER TAB
# ======================================================================
elif active_tab == "🧩 Solver":
    st.header("Reconstruction Solver")

    # --- Settings ---
    with st.expander("⚙️ Solver Settings", expanded=False):
        settings_col1, settings_col2 = st.columns(2)
        with settings_col1:
            model_path = st.text_input(
                "Seam Model File", "data/best_seam_model.pth"
            )
            solver_strategy = st.selectbox(
                "Solver Strategy",
                ["Greedy (Baseline)", "ATSP (Optimized)"],
                index=1,
                help="Greedy: Fast nearest-neighbor chain building. "
                "ATSP: Optimal ordering via Asymmetric TSP solver.",
            )

        with settings_col2:
            confidence_thresh = st.slider(
                "Min Confidence (Greedy)", 0.0, 1.0, 0.5, 0.05
            )
            temperature = st.slider(
                "Temperature (ATSP)",
                0.1,
                2.0,
                0.5,
                0.1,
                help="Lower = sharper score differences. "
                "Controls how aggressively the solver distinguishes good vs bad matches.",
            )

        # Page clustering settings
        st.markdown("---")
        st.markdown("**Page Clustering (Thesis Feature)**")
        cluster_col1, cluster_col2 = st.columns(2)
        with cluster_col1:
            enable_clustering = st.checkbox(
                "Enable HDBSCAN Page Clustering",
                value=False,
                help="Groups strips by page using learned embeddings before ordering. "
                "Requires a trained page embedding model.",
            )
        with cluster_col2:
            if enable_clustering:
                page_model_path = st.text_input(
                    "Page Embedder Model", "data/best_page_embedder.pth"
                )
                min_cluster_size = st.number_input(
                    "Min Cluster Size", min_value=2, value=3
                )

    # --- Clear button ---
    st.markdown('<span class="red-btn-marker"></span>', unsafe_allow_html=True)
    if st.button("Clear Workspace", on_click=clear_workspace):
        pass

    # --- File Upload ---
    uploaded_files = st.file_uploader(
        "Upload Scrambled Strips",
        accept_multiple_files=True,
        type=["jpg", "png", "jpeg"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    shredded_data = st.session_state.get("shredded_data", [])
    input_files = []
    if uploaded_files:
        input_files.extend(uploaded_files)
    if shredded_data:
        st.info(f"Using {len(shredded_data)} strips from Shredder.")
        for item in shredded_data:
            file_obj = io.BytesIO(item["bytes"])
            file_obj.name = item["name"]
            input_files.append(file_obj)

    if input_files:
        # Preview
        with st.expander(
            f"View Input ({len(input_files)} strips)", expanded=False
        ):
            cols = st.columns(min(len(input_files), 8))
            for i, file in enumerate(input_files[:8]):
                cols[i].image(file, caption=f"ID: {i}")
                file.seek(0)

        # --- RECONSTRUCT BUTTON ---
        if st.button("Reconstruct Document", type="primary"):
            model, device = load_seam_model(model_path)

            if not model:
                st.error(f"Model '{model_path}' not found!")
            else:
                with st.spinner("Analyzing text patterns..."):
                    images_pil, images_cv = process_uploads(input_files)
                    n = len(images_pil)

                    # Convert to grayscale for robust inference
                    images_inference = [to_grayscale_rgb(img) for img in images_pil]

                    # --- Compute Score Matrix ---
                    progress_bar = st.progress(0)

                    def update_progress(frac):
                        progress_bar.progress(frac)

                    score_matrix, logit_matrix = compute_score_matrix(
                        model, device, images_inference,
                        progress_callback=update_progress,
                    )

                    # --- Phase 1: Page Clustering (optional) ---
                    clusters = None
                    if enable_clustering:
                        page_model, _ = load_page_embedder(
                            page_model_path, device
                        )
                        if page_model is None:
                            st.warning(
                                "Page embedder not found. "
                                "Running without clustering."
                            )
                        else:
                            with st.spinner("Clustering strips by page..."):
                                # Compute embeddings
                                embeddings = []
                                for img in images_inference:
                                    tensor = preprocess_single_strip(img)
                                    tensor = tensor.unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        emb = page_model(tensor)
                                    embeddings.append(
                                        emb.cpu().numpy().squeeze()
                                    )
                                embeddings = np.array(embeddings)

                                # Cluster
                                try:
                                    from src.solver.clustering import (
                                        cluster_strips_by_page,
                                        detect_and_split_merged_clusters,
                                    )

                                    clusters = cluster_strips_by_page(
                                        embeddings,
                                        min_cluster_size=min_cluster_size,
                                        score_matrix=score_matrix,
                                    )
                                    st.info(
                                        f"📊 HDBSCAN found "
                                        f"{len(clusters)} page clusters"
                                    )

                                    # Block detection: split merged clusters
                                    refined_clusters = {}
                                    cluster_id = 0
                                    for _, indices in clusters.items():
                                        splits = detect_and_split_merged_clusters(
                                            indices, score_matrix
                                        )
                                        if splits is not None:
                                            for sub in splits:
                                                refined_clusters[cluster_id] = sub
                                                cluster_id += 1
                                        else:
                                            refined_clusters[cluster_id] = indices
                                            cluster_id += 1
                                    clusters = refined_clusters
                                    st.info(
                                        f"After block detection: "
                                        f"{len(clusters)} clusters"
                                    )

                                except ImportError as e:
                                    st.warning(
                                        f"Clustering unavailable: {e}. "
                                        f"Running without clustering."
                                    )
                                    clusters = None

                    # --- Phase 2: Solve ---
                    use_atsp = "ATSP" in solver_strategy

                    if clusters is not None:
                        # Solve within each cluster
                        found_pages = []
                        for _, indices in clusters.items():
                            if len(indices) < 2:
                                found_pages.append(indices)
                                continue

                            sub_matrix = score_matrix[
                                np.ix_(indices, indices)
                            ]

                            if use_atsp:
                                sub_logits = logit_matrix[
                                    np.ix_(indices, indices)
                                ]
                                ordering = solve_atsp(
                                    sub_logits,
                                    temperature=temperature,
                                    use_logits=True,
                                )
                            else:
                                pages = solve_greedy(
                                    sub_matrix, threshold=confidence_thresh
                                )
                                ordering = []
                                for p in pages:
                                    ordering.extend(p)

                            # Map back to global indices
                            global_order = [indices[i] for i in ordering]
                            found_pages.append(global_order)
                    else:
                        # No clustering — solve on full pool
                        if use_atsp:
                            ordering = solve_atsp(
                                logit_matrix,
                                temperature=temperature,
                                use_logits=True,
                            )
                            found_pages = [ordering]
                        else:
                            found_pages = solve_greedy(
                                score_matrix, threshold=confidence_thresh
                            )

                    # --- Display Results ---
                    if not found_pages:
                        st.warning("No pages found. Try lowering threshold.")
                    else:
                        st.success(f"Found {len(found_pages)} pages")

                        col_left, col_right = st.columns([1, 3])

                        with col_left:
                            st.markdown("### Score Matrix")
                            fig = px.imshow(
                                score_matrix,
                                color_continuous_scale="RdBu_r",
                                origin="upper",
                            )
                            fig.update_layout(
                                margin=dict(l=0, r=0, t=0, b=0), height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col_right:
                            st.markdown("### Reconstructed Pages")
                            st.caption(
                                "Hover and click the full-screen arrows "
                                "to view details."
                            )

                            for idx, chain in enumerate(found_pages):
                                full_page_bgr = stitch_page(images_cv, chain)
                                st.image(
                                    full_page_bgr,
                                    caption=f"Page {idx + 1} "
                                    f"({len(chain)} strips)",
                                    channels="BGR",
                                    use_container_width=True,
                                )

    else:
        st.info("Upload randomized strips to begin.")
