import torch

def causal_mask(T: int, device=None):
    """Returns a bool mask where True means *masked* (disallowed).
    Shape: (1, 1, T, T) suitable for broadcasting with (B, heads, T, T).
    """
    # A "causal" mask stops token i from attending to any token j > i (the future).
    # Row = query position, column = key position. torch.triu(..., diagonal=1) keeps only the
    # entries strictly ABOVE the diagonal, i.e. exactly the (query, future-key) pairs we must block.
    # e.g. T=3 ->  [[F, T, T],
    #               [F, F, T],
    #               [F, F, F]]
    m = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    # Add two leading singleton dims so the same mask broadcasts across every batch item and head.
    return m.view(1, 1, T, T)