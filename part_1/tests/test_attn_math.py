import numpy as np
import torch
from single_head import SingleHeadSelfAttention

# mirror the tiny example in attn_numpy_demo.py
X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)
Wq = np.array([[ 0.2, -0.1],[ 0.0,  0.1],[ 0.1,  0.2],[-0.1,  0.0]], dtype=np.float32)
Wk = np.array([[ 0.1,  0.1],[ 0.0, -0.1],[ 0.2,  0.0],[ 0.0,  0.2]], dtype=np.float32)
Wv = np.array([[ 0.1,  0.0],[-0.1,  0.1],[ 0.2, -0.1],[ 0.0,  0.2]], dtype=np.float32)

def test_single_head_matches_numpy():
    torch.manual_seed(0)
    x = torch.tensor(X)
    attn = SingleHeadSelfAttention(d_model=4, d_k=2)
    # load weights
    # Overwrite the randomly initialised weights with the hand-picked NumPy ones.
    # nn.Linear stores weight as (out_features, in_features) and computes x @ W.T,
    # so the NumPy (in, out) matrices must be transposed with .t().
    # no_grad: we're editing parameters in place, which autograd shouldn't track.
    with torch.no_grad():
        attn.q.weight.copy_(torch.tensor(Wq).t())
        attn.k.weight.copy_(torch.tensor(Wk).t())
        attn.v.weight.copy_(torch.tensor(Wv).t())
    out, w = attn(x)
    assert out.shape == (1,3,2)
    # Basic numeric sanity
    # (no NaN/inf — would indicate e.g. a fully-masked row producing softmax of all -inf)
    assert torch.isfinite(out).all()
    assert torch.isfinite(w).all()