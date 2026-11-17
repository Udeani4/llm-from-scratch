import torch.nn as nn
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from attn_modern import CausalSelfAttentionModern

class TransformerBlockModern(nn.Module):
    # Same pre-norm residual structure as Part 1/2, but each component can be swapped for its
    # modern (Llama-style) replacement via flags, so you can compare them.
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        # Pick the normalisation class: RMSNorm (modern) or LayerNorm (GPT-2 style).
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln1 = Norm(n_embd)
        self.attn = CausalSelfAttentionModern(n_embd, n_head, dropout, rope, max_pos, sliding_window, attention_sink, n_kv_head)
        self.ln2 = Norm(n_embd)
        # FFN: SwiGLU (gated, modern) or the classic Linear-GELU-Linear MLP.
        self.ffn = SwiGLU(n_embd, mult=4, dropout=dropout) if use_swiglu else nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x, kv_cache=None, start_pos: int = 0):
        # Attention now also returns this layer's updated KV cache, which we pass up to the model.
        a, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + a                          # residual
        x = x + self.ffn(self.ln2(x))      # residual
        return x, kv_cache