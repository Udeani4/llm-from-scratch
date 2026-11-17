import torch
from rmsnorm import RMSNorm

def test_rmsnorm_shapes():
    x = torch.randn(2,3,8)   # (B, T, dim)
    rn = RMSNorm(8)
    y = rn(x)
    # Normalisation is per token and must not change the tensor shape.
    assert y.shape == x.shape