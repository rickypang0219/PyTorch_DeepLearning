# Call Apple Metal Performance Shaders (MPS) backed
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f" Using GPU device: {device}")

# test tensor in MPS
test_tensor = torch.tensor([1, 2, 3]).to(device)
print(test_tensor)
