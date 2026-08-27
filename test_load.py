from datasets import load_dataset
import numpy as np

ds = load_dataset("chainyo/rvl-cdip", split="train")
print("Dataset size:", len(ds))

for i in range(10):
    sample = ds[i]
    img = sample["image"]
    arr = np.array(img.convert("RGB"))
    print("Loaded image", i, "shape:", arr.shape)
