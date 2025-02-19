import torch

# Set device to CUDA
device = torch.device("cuda")

# Initialize tensors on GPU
X = torch.randn(64, 128, device=device) +  torch.randn(64, 128, device=device) * 1.j
W = torch.randn(64, 64, device=device) + torch.randn(64, 64, device=device) * 1.j

# Compute 2D FFT along the last two dimensions
Y = torch.fft.fft(X, dim=(-1))

# Permute to move the 64-dimension to the last position and make contiguous
Y_transformed = Y.permute(1, 0).contiguous() @ W  # Shape: (32, 128, 128, 64) @ (64, 64)

# Compute inverse 2D FFT
Y = torch.fft.ifft(Y_transformed.permute(1, 0).contiguous(), dim=(-1))

# Reference computation without FFT
Y_ref = (X.permute(1, 0).contiguous() @ W).permute(1, 0).contiguous()

# Compare Y and Y_ref
diff = torch.norm(Y - Y_ref) / torch.norm(Y_ref)
print(f"Difference between Y and Y_ref: {diff.item()}")
