import torch
from block_hybrid import HybridFFN

def test_hybrid_ffn_blend():
    B,T,C = 1, 4, 16
    # 30% dense path + 70% MoE path, 3 experts, top-2 routing.
    ffn = HybridFFN(dim=C, alpha=0.3, n_expert=3, k=2)
    x = torch.randn(B,T,C)
    y, aux = ffn(x)
    assert y.shape == x.shape     # must be a drop-in FFN replacement (same in/out shape)
    assert aux.item() >= 0.0