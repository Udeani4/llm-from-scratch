from __future__ import annotations
import torch
import math

# RoPE (Rotary Positional Embedding) — the positional scheme used by Llama, Mistral, Qwen, ...
# Idea: split each q/k vector into pairs of numbers (2D points) and rotate each pair by an angle
#   angle = position * frequency_i
# Different pairs use different frequencies (like the sinusoidal encoding in Part 1).
# Because rotating q by angle(m) and k by angle(n) makes q·k depend only on (m - n),
# attention scores naturally encode RELATIVE position, and nothing is added to the embeddings.

class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim."""
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        assert head_dim % 2 == 0, "RoPE head_dim must be even"   # dims are rotated in pairs
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_pos)
    def get(self, positions: torch.Tensor):
        # positions: (T,) or (1,T)
        # Returns the cos/sin rows for the requested absolute positions.
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            # grow tables
            # (lets generation run past the originally planned max length)
            self._build(max(need, int(self.max_pos * 2)))
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin

    def _build(self, max_pos: int):
        """(Re)build cos/sin tables for a new max_pos."""
        self.max_pos = max_pos
        # One rotation frequency per pair of dims: 1 / 10000^(2i/D). Early pairs rotate fast, later pairs slowly.
        # (Uses the literal 10000.0 rather than self.base.)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        t = torch.arange(max_pos, device=self.device).float()
        # angle[p, i] = p * inv_freq[i]
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    assert x.size(-1) % 2 == 0
    # Add batch and head dims so the same angles apply to every batch item and every head.
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    # Pair up adjacent dims: (x[0], x[1]), (x[2], x[3]), ...
    x1 = x[..., ::2]    # first coord of each pair  (even indices)
    x2 = x[..., 1::2]   # second coord of each pair (odd indices)
    # Standard 2D rotation:  [x1']   [cos -sin] [x1]
    #                        [x2'] = [sin  cos] [x2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    # Interleave the rotated pairs back into their original slots.
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out
