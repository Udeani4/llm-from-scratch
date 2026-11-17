import torch
from attn_mask import causal_mask

def test_mask_is_upper_triangle():
    m = causal_mask(5)
    # ensure shape and diagonal rule
    # (1,1,T,T) so it broadcasts over batch and heads.
    assert m.shape == (1,1,5,5)
    # The number of masked (True) cells must equal the number of strictly-upper-triangular cells,
    # i.e. exactly the "future" positions are blocked and the diagonal (self) is allowed.
    assert m[0,0].sum() == torch.triu(torch.ones(5,5), diagonal=1).sum()
