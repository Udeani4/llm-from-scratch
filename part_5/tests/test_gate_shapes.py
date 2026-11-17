import torch
from gating import TopKGate

def test_gate_topk_shapes():
    # S tokens, C hidden size, E experts, top-K routing
    S, C, E, K = 32, 64, 4, 2
    x = torch.randn(S, C)
    gate = TopKGate(C, E, k=K)
    idx, w, aux = gate(x)
    assert idx.shape == (S, K)   # K expert ids per token
    assert w.shape == (S, K)     # K mixing weights per token
    assert aux.ndim == 0         # load-balancing loss is a single scalar
    # per-token weights are non-negative and <=1
    # (they're softmax probabilities)
    assert torch.all(w >= 0)
    assert torch.all(w <= 1)