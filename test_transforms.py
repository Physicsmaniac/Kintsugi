from PIL import Image
import numpy as np
from torchvision import transforms

img = Image.fromarray(np.zeros((1000, 64, 3), dtype=np.uint8))
t = transforms.Compose([
    transforms.RandomCrop((224, 64), pad_if_needed=True, fill=255),
    transforms.Pad((80, 0, 80, 0), fill=255),
])
out = t(img)
print(out.size)
