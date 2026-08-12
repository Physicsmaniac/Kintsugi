import logging
logging.basicConfig(level=logging.INFO)
from datasets import load_dataset
print("Loading dataset...")
ds = load_dataset("chainyo/rvl-cdip", split="train", streaming=True)
it = iter(ds)
print("Fetching 10 items...")
for i in range(10):
    item = next(it)
    print("Fetched item", i)
print("Done!")
