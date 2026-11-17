from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope_custom import RoPECache, apply_rope_single
from kv_cache import KVCache  # your existing class

# Attention as used in modern LLMs (Llama/Mistral style). Compared to Part 2 it adds:
#   - RoPE: position info injected by rotating q/k (no learned position embedding table)
#   - KV cache: reuse keys/values of past tokens during generation instead of recomputing them
#   - Sliding window + attention sink: bound how many past tokens are attended to (Mistral/StreamingLLM)
#   - GQA (grouped-query attention): fewer K/V heads than Q heads, to shrink the KV cache

class CausalSelfAttentionModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):  # ← NEW
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        # GQA: several query heads share one key/value head. n_kv_head == n_head is ordinary MHA;
        # n_kv_head == 1 is "multi-query attention". Fewer KV heads = smaller KV cache = faster inference.
        self.n_kv_head = n_kv_head or n_head      # ← NEW (GQA defaults to MHA)
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head   # how many Q heads share each KV head
        self.d_head = n_embd // n_head

        # Separate projections for Q vs K/V (sizes differ under GQA)  ← CHANGED
        self.wq  = nn.Linear(n_embd, self.n_head   * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_rope = rope
        self.rope_cache: RoPECache | None = None   # built lazily on first forward (so it lands on the right device)
        self.max_pos = max_pos
        self.sliding_window = sliding_window       # attend to at most this many recent tokens
        self.attention_sink = attention_sink       # ...plus always the first `attention_sink` tokens

    def _maybe_init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """x: (B,T,C). If kv_cache given, we assume generation (T small, often 1)."""
        # start_pos = absolute position of x[:, 0] in the full sequence. With a KV cache we only
        # pass the new token(s), so RoPE needs to know they sit at positions start_pos, start_pos+1, ...
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # Projections
        q = self.wq(x).view(B, T, self.n_head,   self.d_head).transpose(1, 2)    # (B,H, T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)

        # RoPE on *current* tokens (cached keys are already rotated)
        # RoPE rotates q and k by an angle proportional to their position. The dot product q·k then
        # depends only on the RELATIVE distance between the two tokens. V is not rotated.
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope_single(q, cos, sin)   # (B,H, T,D)
            k = apply_rope_single(k, cos, sin)   # (B,Hk,T,D)

        # Concatenate past cache (cache is stored in Hk heads)
        # KV cache: keys/values of earlier tokens never change (causal model), so we store them
        # and just append the new token's k/v. This turns each generation step from O(T) forward
        # work into O(1) new projections.
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Sliding-window + attention-sink (crop along seq length)
        # Keep only the first `s` "sink" tokens + the most recent `sliding_window` tokens.
        # Sink tokens matter because models learn to dump excess attention on the first tokens;
        # dropping them entirely makes long-generation quality collapse (StreamingLLM finding).
        # Note: this crops only what is attended to this step; the full cache is still stored below.
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # --- GQA expand: repeat K/V heads to match Q heads before attention ---
        # e.g. Hk=2, H=4 -> kv heads [a, b] become [a, a, b, b], so Q heads 0,1 use a and 2,3 use b.
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
        else:
            k_attn, v_attn = k_all, v_all

        # Scaled dot-product attention (PyTorch scales internally)
        # No cache (training / first prompt pass): need the causal mask so tokens can't see the future.
        # With cache (one new token): every cached key is in the past, so no mask is needed.
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)          # (B,H,T,D)

        # Merge heads and project back to n_embd.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # Update KV cache (store compact Hk heads, not expanded)
        # Storing only Hk heads is the whole memory saving of GQA.
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk,*,D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)
        # Return the updated cache so the caller can pass it back in on the next generation step.
        return y, new_cache
