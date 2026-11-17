import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class MultiHeadSelfAttention(nn.Module):
    """1.4 Multi-head attention with explicit shape tracing.

    Dimensions (before masking):
      x:      (B, T, d_model)
      qkv:    (B, T, 3*d_model)
      view→   (B, T, 3, n_head, d_head)   where d_head = d_model // n_head
      split→  q,k,v each (B, T, n_head, d_head)
      swap→   (B, n_head, T, d_head)
      scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
      weights:(B, n_head, T, T) = softmax(scores)
      ctx:    (B, n_head, T, d_head) = weights @ v
      merge:  (B, T, n_head*d_head) = (B, T, d_model)
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):
        super().__init__()
        # Each head works on an equal slice of the embedding, so it must divide evenly.
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        # Multiple heads let the model attend to different kinds of relationships in parallel
        # (e.g. one head tracks the previous word, another tracks the subject of the sentence).
        # One fused Linear produces Q, K and V for every head in a single matmul.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection: mixes the concatenated head outputs back into one d_model vector.
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Dropout on the attention weights (randomly drops some token-to-token links during training).
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # (B,T,d_model)
        B, T, C = x.shape
        qkv = self.qkv(x)                          # (B,T,3*C)
        # Carve the 3*C features into (q/k/v, head, per-head dim).
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)  # (B,T,3,heads,dim)
        if self.trace_shapes:
            print("qkv view:", qkv.shape)
        q, k, v = qkv.unbind(dim=2)               # each: (B,T,heads,dim)
        # Put heads before time so each head is processed as an independent batch entry.
        q = q.transpose(1, 2)                      # (B,heads,T,dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.trace_shapes:
            print("q:", q.shape, "k:", k.shape, "v:", v.shape)

        # Similarity of every query with every key, scaled by 1/sqrt(d_head) to keep softmax soft.
        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,heads,T,T)
        # Block attention to future tokens: setting them to -inf makes their softmax weight exactly 0.
        # This is what makes the model "autoregressive" (it can only use the past to predict the next token).
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask, float('-inf'))
        # Turn scores into weights that sum to 1 across keys.
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        # Weighted sum of value vectors = information gathered by each token from the tokens it attends to.
        ctx = torch.matmul(w, v)                  # (B,heads,T,dim)
        if self.trace_shapes:
            print("weights:", w.shape, "ctx:", ctx.shape)
        # Concatenate heads back together: (B,heads,T,dim) -> (B,T,heads*dim).
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,d_model)
        out = self.proj(out)
        if self.trace_shapes:
            print("out:", out.shape)
        # Return weights too so they can be visualised.
        return out, w