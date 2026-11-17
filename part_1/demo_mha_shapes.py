"""Walkthrough of multi-head attention with explicit matrix math and shapes.
Generates a text log at ./out/mha_shapes.txt.
"""
import os
import math
import torch
from multi_head import MultiHeadSelfAttention

OUT_TXT = os.path.join(os.path.dirname(__file__), 'out', 'mha_shapes.txt')


def log(s):
    # Print to console and append the same line to the log file.
    print(s)
    with open(OUT_TXT, 'a') as f:
        f.write(s + "\n")


if __name__ == "__main__":
    # Reset file
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, 'w').close()

    # Tiny config: 1 sequence of 5 tokens, 12-dim embeddings, split into 3 heads of 4 dims each.
    B, T, d_model, n_head = 1, 5, 12, 3
    d_head = d_model // n_head
    # Random stand-in for token embeddings.
    x = torch.randn(B, T, d_model)
    # We reuse the module's learned layers (qkv, proj) but re-do the forward pass by hand below
    # so each intermediate shape can be logged.
    attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=True)

    log(f"Input x:           {tuple(x.shape)} = (B,T,d_model)")
    # One Linear computes Q, K and V for all heads at once (more efficient than 3 separate layers).
    qkv = attn.qkv(x)  # (B,T,3*d_model)
    log(f"Linear qkv(x):     {tuple(qkv.shape)} = (B,T,3*d_model)")

    # Reinterpret the last dim (36) as [3 (q/k/v)] x [3 heads] x [4 dims per head]. No data moves.
    qkv = qkv.view(B, T, 3, n_head, d_head)
    log(f"view to 5D:        {tuple(qkv.shape)} = (B,T,3,heads,d_head)")

    # Split along the "q/k/v" axis into three separate tensors.
    q, k, v = qkv.unbind(dim=2)
    log(f"q,k,v split:       q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    # Move the head axis before T so each head becomes an independent (T, d_head) problem;
    # matmul then treats (B, heads) as batch dimensions.
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    log(f"transpose heads:   q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} = (B,heads,T,d_head)")

    # Scaled dot-product scores: every query vs every key, per head.
    # (No causal mask in this walkthrough — it's purely about shapes.)
    scale = 1.0 / math.sqrt(d_head)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    log(f"scores q@k^T:      {tuple(scores.shape)} = (B,heads,T,T)")

    # Each row becomes a probability distribution over keys.
    weights = torch.softmax(scores, dim=-1)
    log(f"softmax(weights):  {tuple(weights.shape)} = (B,heads,T,T)")

    # Weighted mix of value vectors -> new representation for each token, per head.
    ctx = torch.matmul(weights, v)
    log(f"context @v:        {tuple(ctx.shape)} = (B,heads,T,d_head)")

    # Undo the head split: bring T back to dim 1, then glue heads side-by-side into d_model.
    # .contiguous() is needed because .view() requires memory laid out in order after a transpose.
    out = ctx.transpose(1, 2).contiguous().view(B, T, d_model)
    log(f"merge heads:       {tuple(out.shape)} = (B,T,d_model)")

    # Output projection lets the model mix information across heads.
    out = attn.proj(out)
    log(f"final proj:        {tuple(out.shape)} = (B,T,d_model)")

    log("\nLegend:")
    log("  B=batch, T=sequence length, d_model=embedding size, heads=n_head, d_head=d_model/heads")
    log("  qkv(x) is a single Linear producing [Q|K|V]; we reshape then split into q,k,v")