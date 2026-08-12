import torch
print("Testing CUDA availability:", torch.cuda.is_available())
print("Testing pinned memory allocation...")
try:
    tensor = torch.zeros(1024, 1024).pin_memory()
    print("Pinned memory allocated successfully!")
except Exception as e:
    print("Error:", e)
