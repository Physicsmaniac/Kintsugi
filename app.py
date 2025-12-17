import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import plotly.express as px
import zipfile
import io
import random
import string
import json
import base64
from pdf2image import convert_from_bytes

# --- PAGE CONFIG ---
st.set_page_config(page_title="Forensic Reconstruction AI", layout="wide")

# --- SESSION STATE FOR CLEAR BUTTON ---
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def clear_workspace():
    st.session_state.uploader_key += 1
    if 'shredded_data' in st.session_state:
        del st.session_state['shredded_data']
    if 'current_shred' in st.session_state:
        del st.session_state['current_shred']

def send_to_solver():
    """Callback to transfer shredded data to solver and switch tabs."""
    if 'current_shred' in st.session_state:
        st.session_state['shredded_data'] = st.session_state['current_shred']['generated_strips']
        del st.session_state['current_shred']
        st.session_state.active_tab = "🧩 Solver"

# --- CONSTANTS ---
TRAINING_STRIP_WIDTH = 32
CROP_SIZE = 224

# --- MODEL DEFINITION ---
class SeamResNet(nn.Module):
    def __init__(self):
        super(SeamResNet, self).__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, x): return self.cnn(x)

# --- UTILS ---
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeamResNet()
    if not os.path.exists(model_path):
        return None, device
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

def process_uploads(uploaded_files):
    images_pil = []
    images_cv = []
    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        images_cv.append(img_cv)
        images_pil.append(img_pil)
    return images_pil, images_cv

# --- SOLVER LOGIC ---
def solve_mixed_bag(score_matrix, n, threshold):
    status_container = st.empty()
    status_container.text(f"Sorting {n} mixed strips into pages...")

    visited_global = set()
    pages = []

    while len(visited_global) < n:
        best_path = []
        best_avg_score = -1.0
        available_starts = [i for i in range(n) if i not in visited_global]

        if not available_starts: break

        for start_node in available_starts:
            current_path = [start_node]
            current_visited = {start_node}
            current_score_sum = 0.0
            curr = start_node

            while True:
                row = score_matrix[curr].copy()
                mask_indices = list(visited_global.union(current_visited))
                if mask_indices: row[mask_indices] = -1.0

                next_node = np.argmax(row)
                confidence = row[next_node]

                if confidence < threshold: break

                current_path.append(next_node)
                current_visited.add(next_node)
                current_score_sum += confidence
                curr = next_node

            if len(current_path) > 1:
                avg_score = current_score_sum / (len(current_path) - 1)
                score_metric = avg_score * (len(current_path) ** 0.5)
            else:
                score_metric = 0

            if score_metric > best_avg_score:
                best_avg_score = score_metric
                best_path = current_path

        if len(best_path) >= 2:
            pages.append(best_path)
            visited_global.update(best_path)
        else:
            break

    status_container.empty()
    return pages

# --- UI LAYOUT ---
st.title("Forensic Document Reconstructor")

# Custom CSS for Green Primary Buttons
st.markdown("""
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
    
    /* Red Button Styling (Targeted by marker) */
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
""", unsafe_allow_html=True)

# --- NAVIGATION ---
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🧩 Solver"

# Custom Button-based Navigation
nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    if st.button("🧩 Solver", use_container_width=True, type="primary" if st.session_state.active_tab == "🧩 Solver" else "secondary"):
        st.session_state.active_tab = "🧩 Solver"
        st.rerun()

with nav_col2:
    if st.button("📄 Shredder", use_container_width=True, type="primary" if st.session_state.active_tab == "📄 Shredder" else "secondary"):
        st.session_state.active_tab = "📄 Shredder"
        st.rerun()

active_tab = st.session_state.active_tab

if active_tab == "📄 Shredder":
    st.header("Document Shredder")
    st.markdown("Upload a PDF or Image to shred into randomized strips.")

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "jpg", "png", "jpeg"])
    
    col1, col2 = st.columns(2)
    with col1:
        is_pdf = uploaded_file is not None and uploaded_file.type == "application/pdf"
        page_num = st.number_input("Page Number (PDF only)", min_value=1, value=1, disabled=not is_pdf)
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
                        # Convert PDF bytes to images
                        pdf_bytes = uploaded_file.read()
                        images = convert_from_bytes(
                            pdf_bytes,
                            dpi=300,
                            first_page=page_num,
                            last_page=page_num
                        )
                        if not images:
                            st.error(f"Page {page_num} not found in PDF.")
                        else:
                            page_img = np.array(images[0])
                    else:
                        # Handle Image
                        image = Image.open(uploaded_file).convert('RGB')
                        page_img = np.array(image)

                    if page_img is not None:
                        # Convert RGB to BGR for OpenCV if needed
                        if page_img.ndim == 3:
                            page_img = cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR)
                            
                        height, width, _ = page_img.shape
                        strip_width = width // num_strips
                        
                        # Prepare ZIP in memory
                        zip_buffer = io.BytesIO()
                        ground_truth = {}
                        generated_strips = [] # Store for direct solver usage
                        
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for i in range(num_strips):
                                x_start = i * strip_width
                                x_end = width if (i == num_strips - 1) else (i + 1) * strip_width
                                
                                strip = page_img[:, x_start:x_end]
                                
                                # Random ID
                                rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                                filename = f"page_{page_num}_strip_{rand_id}.jpg"
                                
                                # Encode image to bytes
                                is_success, buffer = cv2.imencode(".jpg", strip)
                                if is_success:
                                    img_bytes = buffer.tobytes()
                                    zf.writestr(filename, img_bytes)
                                    generated_strips.append({"name": filename, "bytes": img_bytes})
                                    
                                    # Record Truth
                                    ground_truth[filename] = {
                                        "real_index": i,
                                        "x_start": x_start,
                                        "page": page_num
                                    }
                            
                            # Add Ground Truth
                            zf.writestr("ground_truth.json", json.dumps(ground_truth, indent=4))
                        
                        # Store results in session state
                        st.session_state['current_shred'] = {
                            'page_num': page_num,
                            'num_strips': num_strips,
                            'zip_bytes': zip_buffer.getvalue(),
                            'generated_strips': generated_strips
                        }
                        
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    # Display Results (Persistent)
    if 'current_shred' in st.session_state:
        shred_data = st.session_state['current_shred']
        
        st.success(f"Successfully shredded Page {shred_data['page_num']} into {shred_data['num_strips']} strips!")
        
        # Proof of Randomness
        with st.expander("🕵️ Verify Shredded Strips (Proof of Randomness)", expanded=False):
            st.caption("These are the actual randomized files contained in the ZIP.")
            strips = shred_data['generated_strips']
            
            # Use standard Streamlit columns for a simple, robust grid
            # We'll show them in rows of 10 for smaller images
            cols_per_row = 10
            for i in range(0, len(strips), cols_per_row):
                cols = st.columns(cols_per_row)
                batch = strips[i:i+cols_per_row]
                for j, strip in enumerate(batch):
                    with cols[j]:
                        st.image(strip['bytes'], use_container_width=True)
                        st.markdown(f"<p style='text-align: center; color: #00E5FF; font-size: 9px; font-weight: bold; line-height: 1.1;'>{strip['name']}</p>", unsafe_allow_html=True)
        
        # Layout: Two equal columns, buttons centered within them
        st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
        col1, col2 = st.columns(2)
        
        with col1:
            # Use a container to help with centering if needed, but Streamlit buttons are full width by default in columns
            # To center visually, we can just rely on the column split
            st.download_button(
                label="Download Shredded Strips (ZIP)",
                data=shred_data['zip_bytes'],
                file_name=f"shredded_page_{shred_data['page_num']}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
            
        with col2:
            st.button("Send to Solver ➡️", type="primary", use_container_width=True, on_click=send_to_solver)

elif active_tab == "🧩 Solver":
    st.header("Reconstruction Solver")
    
    # Configuration in Main Area
    with st.expander("Solver Settings", expanded=False):
        model_path = st.text_input("Model File", "best_seam_model_v2.pth")
        confidence_thresh = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
        
    st.markdown('<span class="red-btn-marker"></span>', unsafe_allow_html=True)
    if st.button("Clear Workspace", on_click=clear_workspace):
        pass

    # 1. FILE UPLOAD (Using dynamic key to allow clearing)
    uploaded_files = st.file_uploader(
        "Upload Scrambled Strips",
        accept_multiple_files=True,
        type=['jpg', 'png', 'jpeg'],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # Check for session state data from Shredder
    shredded_data = st.session_state.get('shredded_data', [])
    
    # Combine inputs
    input_files = []
    if uploaded_files:
        input_files.extend(uploaded_files)
    
    if shredded_data:
        st.info(f"Using {len(shredded_data)} strips from Shredder.")
        for item in shredded_data:
            # Create a file-like object
            file_obj = io.BytesIO(item['bytes'])
            file_obj.name = item['name']
            input_files.append(file_obj)

    if input_files:
        # Preview
        with st.expander(f"View Input ({len(input_files)} strips)", expanded=False):
            cols = st.columns(min(len(input_files), 8))
            for i, file in enumerate(input_files[:8]):
                # Handle both UploadedFile and BytesIO
                cols[i].image(file, caption=f"ID: {i}")
                # Reset pointer for processing later
                file.seek(0)

        # 2. RUN BUTTON
        if st.button("Reconstruct Document", type="primary"):
            model, device = load_model(model_path)

            if not model:
                st.error(f"Model '{model_path}' not found!")
            else:
                with st.spinner("Analyzing text patterns..."):
                    images_pil, images_cv = process_uploads(input_files)
                    n = len(images_pil)
                    score_matrix = np.zeros((n, n))

                    # Normalization
                    norm_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

                    progress_bar = st.progress(0)

                    # --- SCORING LOOP ---
                    for i in range(n):
                        batch_crops = []
                        batch_indices = []

                        for j in range(n):
                            if i == j:
                                score_matrix[i, j] = -1.0
                                continue

                            img_a = images_pil[i]
                            img_b = images_pil[j]

                            # PREPROCESSING
                            def resize_to_training_width(img):
                                w, h = img.size
                                if w == 0: return img
                                scale = TRAINING_STRIP_WIDTH / w
                                new_h = int(h * scale)
                                return img.resize((TRAINING_STRIP_WIDTH, new_h), Image.Resampling.BILINEAR)

                            s1 = resize_to_training_width(img_a)
                            s2 = resize_to_training_width(img_b)

                            min_h = min(s1.size[1], s2.size[1])
                            s1 = s1.crop((0, 0, TRAINING_STRIP_WIDTH, min_h))
                            s2 = s2.crop((0, 0, TRAINING_STRIP_WIDTH, min_h))

                            combined = Image.new('RGB', (TRAINING_STRIP_WIDTH * 2, min_h))
                            combined.paste(s1, (0, 0))
                            combined.paste(s2, (TRAINING_STRIP_WIDTH, 0))

                            def pad_to_model_size(img_crop):
                                canvas = Image.new('RGB', (CROP_SIZE, CROP_SIZE), (255, 255, 255))
                                offset_x = (CROP_SIZE - img_crop.size[0]) // 2
                                offset_y = (CROP_SIZE - img_crop.size[1]) // 2
                                canvas.paste(img_crop, (offset_x, offset_y))
                                return canvas

                            crops = []
                            if min_h <= CROP_SIZE:
                                padded = pad_to_model_size(combined)
                                crops = [padded, padded, padded]
                            else:
                                c1 = combined.crop((0, 0, TRAINING_STRIP_WIDTH*2, CROP_SIZE))
                                mid_y = (min_h - CROP_SIZE) // 2
                                c2 = combined.crop((0, mid_y, TRAINING_STRIP_WIDTH*2, mid_y + CROP_SIZE))
                                c3 = combined.crop((0, min_h - CROP_SIZE, TRAINING_STRIP_WIDTH*2, min_h))
                                crops = [pad_to_model_size(c) for c in [c1, c2, c3]]

                            for c in crops:
                                batch_crops.append(norm_transform(c))
                            batch_indices.append(j)

                        if batch_crops:
                            batch_tensor = torch.stack(batch_crops).to(device)
                            with torch.no_grad():
                                logits = model(batch_tensor)
                                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                            num_neighbors = len(batch_indices)
                            probs_reshaped = probs.reshape(num_neighbors, 3)
                            avg_probs = probs_reshaped.mean(axis=1)

                            for idx, neighbor_idx in enumerate(batch_indices):
                                score_matrix[i, neighbor_idx] = avg_probs[idx]

                        progress_bar.progress((i + 1) / n)

                    # --- SOLVE ---
                    found_pages = solve_mixed_bag(score_matrix, n, confidence_thresh)

                    # --- DISPLAY ---
                    if not found_pages:
                        st.warning("No pages found. Try lowering threshold.")
                    else:
                        st.success(f"Found {len(found_pages)} pages")

                        col_left, col_right = st.columns([1, 3])

                        with col_left:
                             st.markdown("### Matrix")
                             fig = px.imshow(score_matrix, color_continuous_scale='RdBu_r', origin='upper')
                             fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
                             st.plotly_chart(fig, use_container_width=True)

                        with col_right:
                            st.markdown("### Results")
                            st.caption("Hover and click the full-screen arrows to view details.")

                            for idx, chain in enumerate(found_pages):
                                # Stitch
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
                                full_page_bgr = np.concatenate(resized_imgs, axis=1)

                                st.image(
                                    full_page_bgr,
                                    caption=f"Page {idx+1} ({len(chain)} strips)",
                                    channels="BGR",
                                    use_container_width=True
                                )

    else:
        st.info("Upload randomized strips to begin.")
