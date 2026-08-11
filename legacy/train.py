import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import models, transforms
from datasets import load_dataset
import numpy as np
import random
import os
import sys
import time
from PIL import Image, ImageOps
from torch.cuda.amp import GradScaler, autocast

# --- CONFIGURATION ---
# Increase Batch Size if you have multiple GPUs (e.g. 512 for 2 GPUs)
BATCH_SIZE = 256
EPOCHS = 10
STEPS_PER_EPOCH = 3000
VAL_STEPS = 400
LEARNING_RATE = 1e-4

# Worker Config
NUM_WORKERS = min(8, os.cpu_count() or 8)
PREFETCH_FACTOR = 4
STREAM_BUFFER_SIZE = 1000

# --- DATASET: HARD NEGATIVES & RAM SAFE ---
class StreamingShredDataset(IterableDataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        self.split = split
        self.dataset_config = {"path": "chainyo/rvl-cdip", "split": split, "streaming": True}

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = 42 + worker_id
        random.seed(seed)
        np.random.seed(seed)

        dataset = load_dataset(**self.dataset_config)
        if self.split == 'train':
            dataset = dataset.shuffle(seed=seed, buffer_size=1000)

        iterator = iter(dataset)
        buffer = []
        warmup_limit = STREAM_BUFFER_SIZE // 10

        while True:
            try:
                # --- REFILL BUFFER ---
                while len(buffer) < STREAM_BUFFER_SIZE:
                    try:
                        sample = next(iterator)
                        img = sample['image']
                        if img.mode != 'RGB': img = img.convert("RGB")

                        # RAM SAFETY: Resize massive images before storing
                        w, h = img.size
                        if w > 400:
                            new_h = int(h * (400 / w))
                            img = img.resize((400, new_h))

                        buffer.append(np.array(img))
                        if len(buffer) >= warmup_limit: break
                    except StopIteration:
                        iterator = iter(dataset)
                    except:
                        pass

                if not buffer:
                    time.sleep(0.1)
                    continue

                # --- SELECT IMAGE ---
                idx = random.randint(0, len(buffer) - 1)
                img_a = buffer[idx]
                if random.random() < 0.1: buffer.pop(idx) # Keep fresh

                # --- GENERATE PAIR ---
                h, w, c = img_a.shape
                if w < 60: continue

                strip_width = max(32, w // 20)

                choice = random.random()

                # 1. POSITIVE (50%)
                if choice < 0.5:
                    if w - (2 * strip_width) <= 0: continue
                    start_x = random.randint(0, w - (2 * strip_width))
                    strip1 = img_a[:, start_x : start_x + strip_width]
                    strip2 = img_a[:, start_x + strip_width : start_x + (2 * strip_width)]
                    label = 1.0

                # 2. HARD NEGATIVE - VERTICAL SHIFT (25%)
                elif choice < 0.75:
                    if w - (2 * strip_width) <= 0: continue
                    start_x = random.randint(0, w - (2 * strip_width))
                    strip1 = img_a[:, start_x : start_x + strip_width]

                    neighbor = img_a[:, start_x + strip_width : start_x + (2 * strip_width)]

                    # Shift up/down by 5-30 pixels to trap the AI
                    shift = random.choice([-30, -20, -10, 10, 20, 30])
                    n_h, n_w, _ = neighbor.shape
                    n_pil = Image.fromarray(neighbor)
                    canvas = Image.new("RGB", (n_w, n_h), (255, 255, 255))
                    canvas.paste(n_pil, (0, shift))

                    strip2 = np.array(canvas)
                    label = 0.0

                # 3. EASY NEGATIVE (25%)
                else:
                    if w - strip_width <= 0: continue
                    x_a = random.randint(0, w - strip_width)
                    x_b = (x_a + random.randint(strip_width, w)) % (w - strip_width)
                    strip1 = img_a[:, x_a : x_a + strip_width]
                    strip2 = img_a[:, x_b : x_b + strip_width]
                    label = 0.0

                combined = np.concatenate((strip1, strip2), axis=1)
                img_final = Image.fromarray(combined)

                if self.transform:
                    img_final = self.transform(img_final)

                yield img_final, torch.tensor([label], dtype=torch.float32)

            except Exception:
                continue

# --- MODEL DEFINITION (FIXED INDENTATION) ---
class SeamResNet(nn.Module):
    def __init__(self):
        super(SeamResNet, self).__init__()
        # self.cnn MUST be attached to self to be seen by optimizer
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.cnn(x)

# --- TRAINING LOOP ---
def train():
    # 1. TRANSFORMS (Crops, not Resize)
    train_transform = transforms.Compose([
        transforms.RandomCrop((224, 224), pad_if_needed=True, fill=255),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.RandomCrop((224, 224), pad_if_needed=True, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"🚀 Launching Training ({NUM_WORKERS} workers)...")

    train_ds = StreamingShredDataset("train", train_transform)
    val_ds = StreamingShredDataset("val", val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=min(4, NUM_WORKERS), pin_memory=True)

    # 2. SETUP MODEL & HARDWARE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Init Model
    model = SeamResNet()

    # Load Weights BEFORE Multi-GPU wrapper
    if os.path.exists("best_seam_model.pth"):
        print("🔄 Resuming from checkpoint...")
        try:
            state = torch.load("best_seam_model.pth", map_location='cpu')
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state)
            print("✅ Loaded.")
        except:
            print("⚠️ Checkpoint ignored (start fresh).")

    model = model.to(device)

    # Multi-GPU Logic
    if torch.cuda.device_count() > 1:
        print(f"⚡ Multi-GPU Active: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer (Now parameters are guaranteed to exist)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0

    # 3. LOOP
    for epoch in range(EPOCHS):
        model.train()
        print(f"\n📢 Epoch {epoch+1}/{EPOCHS}")

        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            # FIXED AUTOCAST SYNTAX
            with autocast(dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            if i % 50 == 0 and i > 0:
                speed = (i * BATCH_SIZE) / (time.time() - start_time)
                acc = 100 * correct / total
                print(f"   Step {i}/{STEPS_PER_EPOCH} | Loss: {loss.item():.4f} | Acc: {acc:.1f}% | Speed: {speed:.0f} img/s")

            if i >= STEPS_PER_EPOCH: break

        # VALIDATION
        print("🔍 Validating...")
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(val_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                # FIXED AUTOCAST SYNTAX
                with autocast(dtype=torch.float16):
                    outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                if i >= VAL_STEPS: break

        val_acc = 100 * val_correct / val_total
        print(f"✅ Epoch {epoch+1} Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_seam_model.pth")
            print("🏆 Saved Best Model.")

if __name__ == "__main__":
    train()
