from datasets import load_dataset
import numpy as np

# Load in streaming mode to bypass disk limit locally
ds = load_dataset("chainyo/rvl-cdip", split="train", streaming=True)

it = iter(ds)
print("Started iterating")
for i in range(5):
    sample = next(it)
    print("Got sample", i)
    img = sample["image"]
    arr = np.array(img.convert("RGB"))
    print("Decoded sample", i, arr.shape)
