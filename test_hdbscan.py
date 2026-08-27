import torch
import numpy as np
import hdbscan
from PIL import Image
from scripts.benchmark import get_models
from src.data.preprocessing import preprocess_single_strip

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, page_model = get_models("checkpoints/seam/best_seam_model.pth", "checkpoints/embedder/best_page_embedder.pth", device)
    
    # create some dummy images
    img1 = Image.new('RGB', (32, 1000), (255, 0, 0))
    img2 = Image.new('RGB', (32, 1000), (0, 255, 0))
    
    t1 = preprocess_single_strip(img1).unsqueeze(0).to(device)
    t2 = preprocess_single_strip(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        e1 = torch.nn.functional.normalize(page_model(t1), p=2, dim=1).cpu().numpy()
        e2 = torch.nn.functional.normalize(page_model(t2), p=2, dim=1).cpu().numpy()
        
    print("e1 shape:", e1.shape)
    print("e1 norm:", np.linalg.norm(e1))
    print("Cosine sim:", np.dot(e1[0], e2[0]))

if __name__ == '__main__':
    test()
