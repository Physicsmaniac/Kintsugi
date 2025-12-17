import os
import shutil
import cv2
import numpy as np
import random
import string
import json
from pdf2image import convert_from_path

# --- CONFIGURATION ---
SOURCE_PDF = "enron2001.pdf"   # Put your PDF file name here
PAGE_TO_SHRED = 49                  # Page number (1-based)
NUM_STRIPS = 20                    # How many strips?
OUTPUT_DIR = "shredded_randomized" # Output folder
DPI = 300                          # Keep high for crisp text

def generate_random_id(length=8):
    """Generates a random alphanumeric string."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def shred_and_randomize():
    # 1. Setup Output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📄 Processing '{SOURCE_PDF}', Page {PAGE_TO_SHRED}...")

    # 2. Convert PDF to Image
    try:
        images = convert_from_path(
            SOURCE_PDF,
            dpi=DPI,
            first_page=PAGE_TO_SHRED,
            last_page=PAGE_TO_SHRED
        )
        if not images:
            print(f"❌ Error: Page {PAGE_TO_SHRED} not found.")
            return

        # Convert to OpenCV format (BGR)
        page_img = np.array(images[0])
        if page_img.ndim == 3:
            page_img = cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return

    # 3. Shred
    height, width, _ = page_img.shape
    strip_width = width // NUM_STRIPS

    print(f"   Original Size: {width}x{height}")
    print(f"   Shredding into {NUM_STRIPS} strips...")

    ground_truth = {}  # Dictionary to store the correct order

    for i in range(NUM_STRIPS):
        # Calculate coordinates
        x_start = i * strip_width
        x_end = width if (i == NUM_STRIPS - 1) else (i + 1) * strip_width

        # Cut the strip
        strip = page_img[:, x_start:x_end]

        # --- RANDOMIZE NAME ---
        rand_id = generate_random_id()
        filename = f"page_{PAGE_TO_SHRED}_strip_{rand_id}.jpg"

        # Save Image
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, strip)

        # Record the TRUTH (Filename -> Correct Index)
        ground_truth[filename] = {
            "real_index": i,
            "x_start": x_start
        }

        # print(f"   ✂️ Saved {filename} (Real Index: {i})")

    # 4. Save Ground Truth (Hidden Answer Key)
    json_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(json_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)

    print(f"\n✅ Done! Strips saved to '{OUTPUT_DIR}'")
    print(f"🤫 Answer key saved to '{json_path}' (Don't let the AI see it!)")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_PDF):
        print(f"❌ File '{SOURCE_PDF}' not found! Please edit the CONFIG at the top.")
    else:
        shred_and_randomize()
