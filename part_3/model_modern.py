from __future__ import annotations
import torch
import torch.nn as nn
from block_modern import TransformerBlockModern
from tokenizer import ByteTokenizer

# Get the absolute path to the folder that contains part_2 and part_3
# (lets later parts import this file from part_3 by adding the repo root to the import path)
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

class GPTModern(nn.Module):
    # Llama-style GPT. Reused by later parts (4, 6, 7, 8, 9) as the base language model.
    def __init__(self, vocab_size: int = 256, block_size: int = 256,
                 n_layer: int=4, n_head: int=4, n_embd: int=256, dropout: float=0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True, rope: bool = True,
                 max_pos: int = 4096, sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # self.pos_emb = nn.Embedding(block_size, n_embd)
        # ^ No learned position table: RoPE inside attention now supplies position information.
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockModern(n_embd, n_head, dropout, use_rmsnorm, use_swiglu, rope, max_pos, sliding_window, attention_sink, n_kv_head)
            for _ in range(n_layer)
        ])
        # Final norm before the LM head. (With RMSNorm enabled this repo skips it — an Identity no-op;
        # real Llama models do apply a final RMSNorm here.)
        self.ln_f = nn.Identity() if use_rmsnorm else nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)   # hidden -> next-token logits

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_cache_list=None, start_pos: int = 0):
        # kv_cache_list: one KVCache per layer (or None for a fresh, uncached pass).
        # start_pos: absolute position of idx[:, 0], needed by RoPE when only new tokens are fed in.
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)   # (unused now that pos_emb is gone)
        x = self.tok_emb(idx)
        # + self.pos_emb(pos)
        x = self.drop(x)

        new_caches = []
        for i, blk in enumerate(self.blocks):
            # Each layer has its own cache (its own keys/values), so pick layer i's entry.
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache = blk(x, kv_cache=cache, start_pos=start_pos)
            new_caches.append(cache)
        x = self.ln_f(x)
        logits = self.head(x)   # (B,T,vocab)

        loss = None
        if targets is not None:
            import torch.nn.functional as F
            # Standard next-token cross-entropy (same as Part 2).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # Also return the caches so generate() can reuse them on the next step.
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self,
                 prompt: torch.Tensor,
                 max_new_tokens=200,
                 temperature=1.0,
                 top_k=50,
                 top_p=None,
                 eos_id=1, # addition from part 6 for early stopping
                 sliding_window: int | None = None,
                 attention_sink: int = 0):
        # Fast generation using the KV cache.
        # (sliding_window/attention_sink params are unused here — those are set on the attention layers at construction.)
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x   # fallback: no filtering if utils isn't importable

        self.eval()
        idx = prompt
        kvs = [None] * len(self.blocks)   # start with empty caches for every layer

        for _ in range(max_new_tokens):
            # feed full prompt once; then only the last token
            # First step ("prefill"): process the whole prompt and fill the caches.
            # Later steps ("decode"): only the newly generated token needs to be run through the model.
            idx_cond = idx[:, -self.block_size:] if kvs[0] is None else idx[:, -1:]

            # absolute start position from cache length (0 on first step)
            start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

            logits, _, kvs = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)

            # Same sampling recipe as Part 2: last position, temperature, top-k/top-p, softmax.
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            # temperature 0 -> greedy (argmax); otherwise sample randomly from the distribution.
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

            # addition from part 6 for early stopping
            # Stop once every sequence in the batch has produced the end-of-sequence token.
            if eos_id is not None:
                if (next_id == eos_id).all():
                    break

        return idx


    @torch.no_grad()
    def generate_nocache(self, prompt: torch.Tensor, max_new_tokens=200, temperature=1.0, top_k=50, top_p=None,
                sliding_window: int | None = None, attention_sink: int = 0):
        # Reference (slow) generation: re-runs the whole context every step. Used to check that the
        # cached version produces the same output and to compare speed.
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        self.eval()
        idx = prompt

        for _ in range(max_new_tokens):
            # always run a full forward over the cropped window, with NO cache
            idx_cond = idx[:, -self.block_size:]
            # absolute position of first token in the window (matches cached path)
            start_pos = idx.size(1) - idx_cond.size(1)

            logits, _, _ = self(idx_cond, kv_cache_list=None, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            # Debug prints: the 10 most likely next tokens and their probabilities at this step.
            topv, topi = torch.topk(probs, 10)
            print("top ids:", topi.tolist())
            print("top vs:", topv.tolist())
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

