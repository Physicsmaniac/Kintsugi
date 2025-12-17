import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import glob
import shutil
import sys

# --- CONFIG ---
MODEL_PATH = "best_seam_model_v2.pth"
TEST_DIR = "shredded_randomized"
OUTPUT_DIR = "final_mixed_reconstruction"
BATCH_SIZE = 64
CONFIDENCE_THRESHOLD = 0.5   # Connections weaker than this break the chain
TRAINING_STRIP_WIDTH = 32    # Matches training resolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL (Same as before) ---
class SeamResNet(nn.Module):
    def __init__(self):
        super(SeamResNet, self).__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, x): return self.cnn(x)

def load_model():
    model = SeamResNet()
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}"); sys.exit(1)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# --- PREPROCESSING (Same as before) ---
norm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class InferenceDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.pairs = []
        n = len(images)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                self.pairs.append((i, j))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        img_a = self.images[i]
        img_b = self.images[j]

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

        return torch.stack([norm_transform(c) for c in crops]), i, j

# --- NEW: MULTI-PAGE SOLVER ---
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

def main():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(f"{TEST_DIR}/*.jpg") + glob.glob(f"{TEST_DIR}/*.png"))
    n = len(files)
    if n == 0: print("No images found!"); return

    print(f"📦 Loaded {n} strips (Mixed Bag).")
    images_pil = [Image.open(f).convert("RGB") for f in files]
    images_cv = [cv2.imread(f) for f in files]

    model = load_model()
    score_matrix = np.zeros((n, n))
    dataset = InferenceDataset(images_pil)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    print("⚡ Computing all-to-all scores...")
    with torch.no_grad():
        for batch_crops, idx_i, idx_j in loader:
            b, num_crops, c, h, w = batch_crops.shape
            batch_flat = batch_crops.view(-1, c, h, w).to(device)
            logits = model(batch_flat)
            probs = torch.sigmoid(logits).view(b, num_crops)
            avg_probs = probs.mean(dim=1).cpu().numpy()

            for k, prob in enumerate(avg_probs):
                r, c = idx_i[k].item(), idx_j[k].item()
                score_matrix[r, c] = prob

    # --- SOLVE MULTIPLE PAGES ---
    pages = solve_mixed_bag(score_matrix, n)

    print(f"🎉 Identified {len(pages)} distinct pages.")

    for i, path in enumerate(pages):
        ordered_imgs = [images_cv[k] for k in path]
        target_h = min(im.shape[0] for im in ordered_imgs)
        resized_imgs = []
        for img in ordered_imgs:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            resized_imgs.append(cv2.resize(img, (new_w, target_h)))

        full_stitch = np.concatenate(resized_imgs, axis=1)
        filename = f"{OUTPUT_DIR}/reconstructed_page_{i+1}.jpg"
        cv2.imwrite(filename, full_stitch)
        print(f"  💾 Saved {filename}")

if __name__ == "__main__":
    main()
