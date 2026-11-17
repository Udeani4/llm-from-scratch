"""Visualize multi-head attention weights per head (grid)."""
import torch
from multi_head import MultiHeadSelfAttention
from vis_utils import save_attention_heads_grid

B, T, d_model, n_head = 1, 5, 12, 3
x = torch.randn(B, T, d_model)  # random "token embeddings"
attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=False)

# w holds each head's attention pattern: w[0, h, i, j] = how much token i looks at token j in head h.
out, w = attn(x)  # w: (B, H, T, T)

# .detach() drops the autograd graph, .cpu().numpy() converts for matplotlib.
# Expect lower-triangular heatmaps because of the causal mask.
save_attention_heads_grid(w.detach().cpu().numpy(), filename="multi_head_attn_grid.png")