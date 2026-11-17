from __future__ import annotations
import torch, torch.nn as nn

class RewardModel(nn.Module):
    """Transformer encoder → pooled representation → scalar reward.
    Bidirectional encoder is fine for reward modeling (not used for generation).
    """
    # A reward model reads a whole (prompt + response) and outputs ONE number: how good a human
    # would judge it. It's a learned stand-in for a human rater, used in Parts 8-9 to score the
    # policy's outputs during RL. (Production RMs are usually the SFT LLM itself with its LM head
    # swapped for a 1-output head; here it's a separate small encoder for simplicity.)
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)   # learned absolute positions
        # PyTorch's built-in transformer encoder layer: attention + FFN, with NO causal mask —
        # every token can see every other token, which is fine since we don't generate with it.
        enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,
                                               dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embd)
        # Regression head: hidden vector -> single scalar score.
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor):
        # x: (B,T) token ids, padded with id 2. Returns (B,) rewards.
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        # True where the token is padding; attention will ignore those keys.
        pad_mask = (x == 2)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        # masked mean pool over tokens (ignoring pads)
        # Collapse the T per-token vectors into one sequence vector by averaging the real tokens.
        # (LLM-based RMs instead take the hidden state of the LAST token.)
        mask = (~pad_mask).float().unsqueeze(-1)     # (B,T,1): 1 for real tokens, 0 for pads
        h_sum = (h * mask).sum(dim=1)                # (B,C)
        len_ = mask.sum(dim=1).clamp_min(1.0)        # number of real tokens (avoid /0)
        pooled = h_sum / len_
        r = self.head(pooled).squeeze(-1)  # (B,)
        return r