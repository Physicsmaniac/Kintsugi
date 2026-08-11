import fitz  # pymupdf
import cv2
import numpy as np
import os
import random
import shutil
import json

# --- CONFIG ---
INPUT_PDF = "206-10001-10017.pdf"   # <--- RENAME THIS to your PDF file
OUTPUT_DIR = "person5_test_set/tagged_chaos"
NUM_STRIPS_PER_PAGE = 10
MAX_PAGES = 10                    # <--- HOW MANY PAGES TO SHRED

def shred_subset():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc = fitz.open(INPUT_PDF)

    # Safety check: Don't crash if PDF is shorter than MAX_PAGES
    pages_to_process = min(len(doc), MAX_PAGES)
    print(f"🔥 Shredding first {pages_to_process} pages from {INPUT_PDF}...")

    strips_metadata = []

    for page_num in range(pages_to_process):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200) # 200 DPI is standard fax quality

        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_data.reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w, c = img.shape
        strip_width = w // NUM_STRIPS_PER_PAGE

        for i in range(NUM_STRIPS_PER_PAGE):
            x_start = i * strip_width
            x_end = (i + 1) * strip_width if i < NUM_STRIPS_PER_PAGE - 1 else w
            strip = img[:, x_start:x_end]

            strips_metadata.append({
                "image": strip,
                "original_page": page_num,
                "original_index": i
            })

        print(f"   Processed Page {page_num + 1}")

    # SHUFFLE
    print(f"🌪️  Mixing {len(strips_metadata)} strips...")
    random.shuffle(strips_metadata)

    # SAVE
    final_ground_truth = {}

    for idx, item in enumerate(strips_metadata):
        fname = f"image_{idx:04d}.jpg"
        cv2.imwrite(f"{OUTPUT_DIR}/{fname}", item['image'])

        final_ground_truth[fname] = {
            "page": item['original_page'],
            "index": item['original_index']
        }

    with open(f"{OUTPUT_DIR}/ground_truth.json", "w") as f:
        json.dump(final_ground_truth, f, indent=2)

    print(f"✅ Done. Saved test set in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    shred_subset()
