import logging
logging.basicConfig(level=logging.INFO)
from datasets import load_dataset
print("Loading dataset...")
ds = load_dataset("chainyo/rvl-cdip", split="train", streaming=True)
print("Dataset loaded. Getting iterator...")
it = iter(ds)
print("Iterator created. Fetching first item...")
try:
    item = next(it)
    print("Successfully fetched item:", item.keys())
except Exception as e:
    print("Error:", e)
