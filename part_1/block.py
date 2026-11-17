import torch.nn as nn
from multi_head import MultiHeadSelfAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    """1.6 Transformer block = LN → MHA → residual → LN → FFN → residual."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        # LayerNorm normalises each token's vector (zero mean, unit variance, + learned scale/shift).
        # Keeps activations well-scaled so deep stacks of blocks train stably.
        self.ln1 = nn.LayerNorm(d_model)
        # Attention = the step where tokens exchange information with each other.
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        # FFN = per-token "thinking" step; no mixing across positions happens here.
        self.ffn = FeedForward(d_model, mult=4, dropout=dropout)

    def forward(self, x):
        # "Pre-LN" design: normalise BEFORE each sub-layer, then add the result back to x.
        # The `x + ...` residual connection gives gradients a direct path through the network,
        # which is what makes stacking dozens of blocks trainable.
        # [0] because our attention returns (output, attention_weights).
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x