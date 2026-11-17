"""1.1 Positional encodings (absolute learned + sinusoidal)."""
import math
import torch
import torch.nn as nn

# Why positional encodings? Attention itself is order-blind: shuffling the tokens just shuffles the
# output. To let the model know WHERE each token is, we add a position-dependent vector to each
# token embedding.

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # A lookup table with one trainable vector per position 0..max_len-1 (GPT-2 style).
        # Downside: can't handle sequences longer than max_len.
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model) — we only need its T and device
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)  # [0, 1, ..., T-1]
        pos_emb = self.emb(pos)  # (T, d_model)
        # Add (not concatenate) the position vector to each token's embedding.
        return x + pos_emb.unsqueeze(0)  # broadcast over batch

class SinusoidalPositionalEncoding(nn.Module):
    # Fixed (non-learned) encoding from the original "Attention Is All You Need" paper.
    # Each pair of dimensions is a sine/cosine wave of a different frequency, so every position
    # gets a unique pattern and relative offsets correspond to simple rotations.
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        # Column vector of positions, shape (max_len, 1).
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Frequencies: 1 / 10000^(2i/d_model) for i = 0, 1, ... (computed in log space for stability).
        # Low dims oscillate fast (fine position detail), high dims oscillate slowly (coarse position).
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Even dims get sin, odd dims get cos. position * div_term broadcasts to (max_len, d_model/2).
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer: saved with the model & moved by .to(device), but NOT a trainable parameter.
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        # Take the first T position vectors and add them to every sequence in the batch.
        return x + self.pe[:T].unsqueeze(0)