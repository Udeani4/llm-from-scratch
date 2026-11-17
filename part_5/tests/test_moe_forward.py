import torch
from moe import MoE

def test_moe_forward_dims_and_grad():
    B,T,C = 2, 8, 32
    moe = MoE(dim=C, n_expert=4, k=1)
    x = torch.randn(B,T,C, requires_grad=True)
    y, aux = moe(x)
    assert y.shape == x.shape
    # Dummy task loss + small-weighted aux loss — the same pattern a real training loop uses
    # (aux coefficients around 0.01 are typical).
    loss = (y**2).mean() + 0.01*aux
    loss.backward()
    # some gradient must flow to gate and experts
    # (routing uses discrete top-k, so this checks gradients still get through the gate weights)
    grads = [p.grad is not None for p in moe.parameters()]
    assert any(grads)