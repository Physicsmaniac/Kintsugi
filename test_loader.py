import logging
logging.basicConfig(level=logging.INFO)
import torch
from torch.utils.data import DataLoader
from src.training.dataset import StreamingShredDataset
from torchvision import transforms

def _train_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

print("Creating dataset...")
ds = StreamingShredDataset(split="train", transform=_train_transforms(), streaming=True)
print("Creating loader...")
loader = DataLoader(ds, batch_size=8, num_workers=0, pin_memory=True)
print("Fetching batch...")
for batch_idx, (images, labels) in enumerate(loader):
    print("GOT BATCH!")
    break
