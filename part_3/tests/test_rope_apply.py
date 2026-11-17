import torch
from rope_custom import RoPECache, apply_rope_single

def test_rope_rotation_shapes_single():
    # Vanilla case: same #heads for q and k
    B, H, T, D = 1, 2, 5, 8
    rc = RoPECache(head_dim=D, max_pos=32)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    pos = torch.arange(0, T)
    cos, sin = rc.get(pos)   # each (T, D/2)

    q2 = apply_rope_single(q, cos, sin)
    k2 = apply_rope_single(k, cos, sin)

    assert q2.shape == q.shape
    assert k2.shape == k.shape
    # check that rotation mixes even/odd features (values should change)
    # (position 0 has angle 0 = no rotation, but positions 1..4 do rotate)
    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)

def test_rope_rotation_shapes_gqa():
    # GQA case: q has H heads; k has fewer Hk heads (shared KV)
    # RoPE only depends on position and head_dim, so the same cos/sin works for any number of heads.
    B, H, Hk, T, D = 2, 8, 2, 7, 16
    rc = RoPECache(head_dim=D, max_pos=128)
    q = torch.randn(B, H,  T, D)
    k = torch.randn(B, Hk, T, D)
    pos = torch.arange(10, 10 + T)  # arbitrary start position
    # (like decoding with a KV cache, where new tokens start at a non-zero position)
    cos, sin = rc.get(pos)

    q2 = apply_rope_single(q, cos, sin)
    k2 = apply_rope_single(k, cos, sin)

    assert q2.shape == (B, H,  T, D)
    assert k2.shape == (B, Hk, T, D)
    # rotations should be deterministic for same positions
    # check a couple of coords moved as expected (values changed)
    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)
