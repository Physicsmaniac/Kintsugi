import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- CONFIG ---
MODEL_PATH = "best_seam_model_v2.pth"
TEST_DIR = "person5_test_set/tagged_chaos"
JSON_FILE = f"{TEST_DIR}/ground_truth.json"
OUTPUT_DIR = "evaluation_results"
CONFIDENCE_THRESHOLD = 0.5
TRAINING_STRIP_WIDTH = 32

# --- MODEL SETUP (Must match training) ---
class SeamResNet(nn.Module):
    def __init__(self):
        super(SeamResNet, self).__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.cnn(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = SeamResNet()
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

        # Fix DataParallel keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        print("❌ Model not found!")
        exit()
    model.to(device)
    model.eval()
    return model

# --- TRANSFORM ---
norm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_pair(img_a, img_b):
    # Resize to training width (32px)
    def resize_to_training_width(img):
        w, h = img.size
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

    # Pad to 224x224
    def pad_to_model_size(img_crop):
        canvas = Image.new('RGB', (224, 224), (255, 255, 255))
        offset_x = (224 - img_crop.size[0]) // 2
        offset_y = (224 - img_crop.size[1]) // 2
        canvas.paste(img_crop, (offset_x, offset_y))
        return canvas

    crop_h = 224
    if min_h <= crop_h:
        padded = pad_to_model_size(combined)
        crops = [padded, padded, padded]
    else:
        c1 = combined.crop((0, 0, TRAINING_STRIP_WIDTH*2, crop_h))
        mid_y = (min_h - crop_h) // 2
        c2 = combined.crop((0, mid_y, TRAINING_STRIP_WIDTH*2, mid_y + crop_h))
        c3 = combined.crop((0, min_h - crop_h, TRAINING_STRIP_WIDTH*2, min_h))
        crops = [pad_to_model_size(c) for c in [c1, c2, c3]]

    return torch.stack([norm_transform(c) for c in crops]).to(device)

def solve_mixed_bag(score_matrix, n):
    """
    Identifies multiple distinct chains (pages) from the pile.
    Returns a LIST of paths (e.g., [[0,1,2], [5,6,7,8], [3,4]])
    """
    print(f"🧠 Sorting {n} mixed strips into pages...")

    visited_global = set()
    pages = []

    # Heuristic: While we still have unvisited nodes...
    while len(visited_global) < n:
        best_path = []
        best_avg_score = -1.0

        # Identify potential start nodes (nodes not yet used)
        available_starts = [i for i in range(n) if i not in visited_global]

        if not available_starts: break

        # Try to build a chain from every available start node
        for start_node in available_starts:
            current_path = [start_node]
            current_visited = {start_node}
            current_score_sum = 0.0
            curr = start_node

            while True:
                # Look at neighbors
                row = score_matrix[curr].copy()

                # Mask globally visited nodes AND current chain nodes
                # (We can't reuse a strip we already pasted in a previous page)
                mask_indices = list(visited_global.union(current_visited))
                row[mask_indices] = -1.0

                next_node = np.argmax(row)
                confidence = row[next_node]

                # --- BREAK CONDITION ---
                # If the best connection is weak, STOP. This is the end of the page.
                if confidence < CONFIDENCE_THRESHOLD:
                    break

                current_path.append(next_node)
                current_visited.add(next_node)
                current_score_sum += confidence
                curr = next_node

            # Score this chain
            if len(current_path) > 1:
                avg_score = current_score_sum / (len(current_path) - 1)
                # Bonus for length (we prefer longer valid pages over 2-strip fragments)
                score_metric = avg_score * (len(current_path) ** 0.5)
            else:
                score_metric = 0 # Single strip is not a page

            if score_metric > best_avg_score:
                best_avg_score = score_metric
                best_path = current_path

        # If we found a valid chain (at least 2 strips), save it as a page
        if len(best_path) >= 2:
            print(f"  📄 Found Page: {best_path} (Score: {best_avg_score:.2f})")
            pages.append(best_path)
            visited_global.update(best_path)
        else:
            # We only have single loose strips left.
            # Mark remaining as "visited" to exit loop or handle as "scraps"
            print(f"  🗑️ Remaining {len(available_starts)} strips could not be confidently attached.")
            break

    return pages

def evaluate_reconstruction():
    model = load_model()

    if not os.path.exists(JSON_FILE):
        print(f"❌ Ground truth file not found: {JSON_FILE}")
        return

    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. LOAD GROUND TRUTH
    with open(JSON_FILE, 'r') as f: gt_data = json.load(f)

    # Group by page
    pages = {}
    for fname, info in gt_data.items():
        p_num = info.get('page', 0)
        # Try to find rank/order
        rank = info.get('index', info.get('rank', info.get('order', info.get('position', -1))))

        # Fallback: Try to parse from filename (e.g., "strip_0.jpg")
        if rank == -1:
            try:
                rank = int(os.path.splitext(fname)[0].split('_')[-1])
            except:
                pass

        if p_num not in pages: pages[p_num] = []
        pages[p_num].append({'file': fname, 'rank': rank})

    print(f"📦 Found {len(pages)} pages in ground truth.")

    total_pairwise_correct = 0
    total_pairs = 0
    all_true_positions = []
    all_pred_positions = []

    # 2. EVALUATE EACH PAGE
    for p_num, strips in pages.items():
        # Sort strips by rank to get True Order
        strips.sort(key=lambda x: x['rank'])

        filenames = [s['file'] for s in strips]
        n = len(filenames)
        if n < 2: continue

        print(f"   🧠 Evaluating Page {p_num} ({n} strips)...")

        # Load Images
        images_pil = []
        valid_filenames = []
        for f in filenames:
            path = os.path.join(TEST_DIR, f)
            if not os.path.exists(path):
                print(f"      ⚠️ Missing file: {f}")
                continue

            img = cv2.imread(path)
            p = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            images_pil.append(p)
            valid_filenames.append(f)

        n = len(images_pil)
        if n < 2: continue

        # Compute Score Matrix
        score_matrix = np.zeros((n, n))
        with torch.no_grad():
            for i in range(n):
                for j in range(n):
                    if i == j:
                        score_matrix[i, j] = -1.0
                        continue
                    tensor = preprocess_pair(images_pil[i], images_pil[j])
                    logits = model(tensor)
                    probs = torch.sigmoid(logits)
                    score_matrix[i, j] = probs.mean().item()

        # --- SOLVER (Mixed Bag Logic) ---
        predicted_pages = solve_mixed_bag(score_matrix, n)
        
        # Flatten the predicted pages into a single path for evaluation
        path = []
        for p in predicted_pages:
            path.extend(p)
            
        # Append any remaining strips that weren't part of a confident chain
        visited_in_path = set(path)
        missing = [x for x in range(n) if x not in visited_in_path]
        path.extend(missing)

        # --- METRICS ---
        # True Order is just 0, 1, 2, ..., n-1 (since we sorted filenames by rank)
        # Predicted Order is 'path' (indices into the sorted list)

        # 1. Pairwise Accuracy (Neighbor Check)
        # A pair (i, i+1) in prediction is correct if it matches (j, j+1) in truth
        # Since truth is [0, 1, 2...], we just check if path[k+1] == path[k] + 1
        page_pairs = 0
        page_correct = 0
        for k in range(len(path) - 1):
            u = path[k]
            v = path[k+1]
            # In ground truth, v should follow u immediately
            # Since we sorted valid_filenames by rank, the index IS the rank.
            # So we expect v == u + 1
            if v == u + 1:
                page_correct += 1
            page_pairs += 1

        if page_pairs > 0:
            acc = page_correct / page_pairs
            print(f"      ✅ Pairwise Accuracy: {acc:.2%}")
            total_pairwise_correct += page_correct
            total_pairs += page_pairs

        # 2. Position Confusion Data
        # True Position of strip 'i' is 'i'
        # Predicted Position of strip 'i' is 'path.index(i)'
        page_true_pos = []
        page_pred_pos = []
        for true_pos in range(n):
            if true_pos in path:
                pred_pos = path.index(true_pos)
                all_true_positions.append(true_pos)
                all_pred_positions.append(pred_pos)
                page_true_pos.append(true_pos)
                page_pred_pos.append(pred_pos)
        
        # Optional: Save per-page confusion matrix
        if len(page_true_pos) > 0:
            max_p = max(max(page_true_pos), max(page_pred_pos)) + 1
            cm_page = confusion_matrix(page_true_pos, page_pred_pos, labels=range(max_p))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_page, annot=True, cmap="Blues", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Page {p_num} Confusion Matrix")
            plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_page_{p_num}.png"))
            plt.close()

    # --- GLOBAL RESULTS ---
    if total_pairs > 0:
        print(f"\n🏆 GLOBAL PAIRWISE ACCURACY: {total_pairwise_correct / total_pairs:.2%}")

    # --- CONFUSION MATRIX ---
    if all_true_positions:
        print("📊 Generating Confusion Matrix...")
        # Cap max size for visualization if needed, or let seaborn handle it
        max_pos = max(max(all_true_positions), max(all_pred_positions)) + 1
        cm = confusion_matrix(all_true_positions, all_pred_positions, labels=range(max_pos))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
        plt.xlabel("Predicted Position")
        plt.ylabel("True Position")
        plt.title("Reconstruction Confusion Matrix (Position vs Position)")

        save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(save_path)
        print(f"💾 Saved Confusion Matrix to {save_path}")

if __name__ == "__main__":
    evaluate_reconstruction()
