import torch
import numpy as np
import time
from src.models.transformer import HybridTransformer  # or whatever model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Dummy inputs
B = 16
seq_len = 200
in_dim = 14
out_dim = 6
input_batch = torch.randn(B, seq_len, in_dim, requires_grad=True).to(device)

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, x):
        # some temporal mixing to make it realistic
        x = x + torch.roll(x, shifts=1, dims=1)
        return self.linear(x)

model = DummyModel().to(device)

t0 = time.time()
outputs = model(input_batch)
out_col = 0
dynamics_maps = np.zeros((B, in_dim, seq_len, seq_len))

for x in range(seq_len):
    if input_batch.grad is not None:
        input_batch.grad.zero_()
    model.zero_grad()
    score_sum = outputs[:, x, out_col].sum()
    is_last = (x == seq_len - 1)
    score_sum.backward(retain_graph=not is_last)
    if input_batch.grad is not None:
        grad_slice = input_batch.grad.cpu().numpy().transpose(0, 2, 1)
        dynamics_maps[:, :, :, x] = np.abs(grad_slice)

t1 = time.time()
print(f"Batched backward time for {B} samples: {t1-t0:.3f}s")
print(f"Output shape: {dynamics_maps.shape}")
