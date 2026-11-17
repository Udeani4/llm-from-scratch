from __future__ import annotations
import torch

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    - logits: (B, vocab)
    Returns filtered logits with -inf for masked entries.
    """
    # Both methods cut off the long tail of unlikely tokens before sampling. Without this,
    # thousands of individually-unlikely tokens add up to a real chance of picking nonsense.
    # Setting a logit to -inf gives it probability exactly 0 after softmax.
    B, V = logits.shape
    filtered = logits.clone()

    # --- top-k: keep only the k highest-scoring tokens ---
    if top_k is not None and top_k < V:
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)   # sorted descending, (B, k)
        kth = topk_vals[:, -1].unsqueeze(-1)                 # the k-th largest value per row = cut-off
        filtered[filtered < kth] = float('-inf')

    # --- top-p (nucleus): keep the smallest set of tokens whose total probability >= p ---
    # Adapts to the model's confidence: few tokens when it's sure, many when it's uncertain.
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)     # running total of probability mass
        mask = cumsum > top_p                    # tokens past the p threshold get dropped
        # keep at least 1 token
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        # Scatter back
        # Put the filtered values back into original vocab order (sorted_idx says where each came from).
        filtered = torch.full_like(filtered, float('-inf'))
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered