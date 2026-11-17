from __future__ import annotations
import torch.nn as nn
from moe import MoE

class HybridFFN(nn.Module):
    """Blend dense FFN with MoE output: y = α * Dense(x) + (1−α) * MoE(x).
    Use α∈[0,1] to trade between stability (dense) and capacity (MoE).
    """
    # Drop-in replacement for a transformer block's FFN. Every token always gets the dense FFN
    # (a reliable "shared" path) plus a routed expert. Similar in spirit to the "shared expert"
    # used by DeepSeek-MoE / Qwen-MoE.
    def __init__(self, dim: int, alpha: float = 0.5, mult: int = 4, swiglu: bool = True, n_expert: int = 4, k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.alpha = alpha   # fixed mixing weight (not learned)
        inner = mult * dim
        # Ordinary dense FFN, applied to every token.
        self.dense = nn.Sequential(
            nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim), nn.Dropout(dropout)
        )
        # Sparse path: each token is routed to k of n_expert expert FFNs.
        self.moe = MoE(dim, n_expert=n_expert, k=k, mult=mult, swiglu=swiglu, dropout=dropout)
    def forward(self, x):
        y_dense = self.dense(x)
        y_moe, aux = self.moe(x)
        y = self.alpha * y_dense + (1.0 - self.alpha) * y_moe
        # Pass the load-balancing loss up so the training loop can add it to the main loss.
        return y, aux