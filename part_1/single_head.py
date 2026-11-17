import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class SingleHeadSelfAttention(nn.Module):
    """1.3 Single-head attention (explicit shapes)."""
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shapes: bool = False):
        super().__init__()
        # Three learned projections (the PyTorch version of Wq/Wk/Wv in attn_numpy_demo.py).
        # bias=False is conventional for attention projections.
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # x: (B, T, d_model)
        B, T, _ = x.shape
        q = self.q(x)  # (B,T,d_k)
        k = self.k(x)  # (B,T,d_k)
        v = self.v(x)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"q {q.shape}  k {k.shape}  v {v.shape}")
        # Scaled dot-product attention: scores[b, i, j] = q_i · k_j / sqrt(d_k).
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,T,T)
        mask = causal_mask(T, device=x.device)
        # Mask is (1,1,T,T) for multi-head use; squeeze the head dim so it broadcasts with (B,T,T).
        # Future positions get -inf -> 0 probability after softmax.
        attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
        w = F.softmax(attn, dim=-1)  # each row sums to 1
        w = self.dropout(w)
        out = torch.matmul(w, v)  # (B,T,d_k)  weighted average of values
        if self.trace_shapes:
            print(f"weights {w.shape}  out {out.shape}")
        return out, w