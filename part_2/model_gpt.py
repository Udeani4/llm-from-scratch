from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Blocks (self-contained for isolation) ----
class CausalSelfAttention(nn.Module):
    # Same idea as part_1/multi_head.py, but the core attention math is delegated to PyTorch's fused kernel.
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # fused Q, K, V projection
        self.proj = nn.Linear(n_embd, n_embd, bias=False)      # mixes heads back together
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):  # (B,T,C)
        B, T, C = x.shape
        # (B,T,3C) -> (B,T,3,heads,d_head) -> three (B,T,heads,d_head) tensors
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # -> (B,heads,T,d_head) so each head is an independent attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_head)  # (unused: SDPA applies this scaling internally)
        # PyTorch SDPA (uses flash when available)
        # scaled_dot_product_attention computes softmax(q@k^T/sqrt(d)) @ v in one optimised kernel.
        # is_causal=True applies the "no looking at future tokens" mask for us.
        # Dropout on attention weights only during training.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        # Merge heads: (B,heads,T,d_head) -> (B,T,C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class FeedForward(nn.Module):
    # Per-token MLP: expand 4x, GELU, project back.
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # One transformer layer (pre-LayerNorm + residual connections).
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # tokens communicate
        x = x + self.ffn(self.ln2(x))    # each token processes what it gathered
        return x

# ---- Tiny GPT ----
class GPT(nn.Module):
    # Full decoder-only language model: token ids in -> scores for the next token at every position out.
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.0):
        super().__init__()
        # block_size = context window: max number of tokens the model can look at at once.
        self.block_size = block_size
        # Token embedding: maps each token id (0..vocab_size-1) to a learned n_embd-dim vector.
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # Learned absolute positional embedding (one vector per position in the context window).
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        # Stack of n_layer identical transformer blocks.
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        # Final LayerNorm before the output layer (needed with pre-LN blocks).
        self.ln_f = nn.LayerNorm(n_embd)
        # "LM head": projects each hidden vector to one raw score (logit) per vocabulary token.
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Apply GPT-2 style initialisation to every submodule.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Small normal init (std 0.02) keeps early activations/logits small so training starts stable.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (B,T) token ids. targets: (B,T) the next-token ids (inputs shifted by 1), or None at inference.
        B, T = idx.shape
        assert T <= self.block_size   # positional table only has block_size entries
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)   # (1,T)
        # Input representation = "what token" + "where it is". Shapes: (B,T,C) + (1,T,C).
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        # logits: (B,T,vocab). logits[b,t] = unnormalised scores for the token at position t+1.
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Cross-entropy = -log(probability the model gave to the correct next token), averaged.
            # Flatten (B,T,V)->(B*T,V) and (B,T)->(B*T) because cross_entropy expects (N,C) and (N).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0,
                top_k: int | None = 50, top_p: float | None = None):
        # Autoregressive generation: predict one token, append it, feed the longer sequence back in, repeat.
        from utils import top_k_top_p_filtering
        self.eval()
        # Guard: if the prompt is empty, start with a newline byte (10)
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens — the model can't see further back than its context window.
            idx_cond = idx[:, -self.block_size:]
            # Full forward pass over the whole context each step (no KV cache yet — that's Part 3).
            logits, _ = self(idx_cond)
            # Only the LAST position's logits predict the next token.
            # Temperature: <1 sharpens the distribution (more deterministic), >1 flattens it (more random).
            # max(..., 1e-6) avoids division by zero when temperature=0.
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            # Remove unlikely tokens (see utils.py) so we don't sample garbage from the long tail.
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)                  # logits -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)      # randomly sample 1 token by those probs
            idx = torch.cat([idx, next_id], dim=1)                 # append and continue
        return idx
