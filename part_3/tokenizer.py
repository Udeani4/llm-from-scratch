from __future__ import annotations
import torch

class ByteTokenizer:
    """Simple byte-level tokenizer (0..255)."""
    # Copy of part_2/tokenizer.py so part_3 is self-contained: each UTF-8 byte is one token.
    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)
    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Ignore invalid byte sequences the model may produce.
        return bytes(ids).decode('utf-8', errors='ignore')
    @property
    def vocab_size(self) -> int:
        return 256