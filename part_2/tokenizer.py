from __future__ import annotations
import torch

# A tokenizer converts text <-> integer ids, because neural nets operate on numbers.
# This one uses raw UTF-8 bytes as tokens: tiny vocabulary (256), can encode ANY text,
# but sequences are long (1 token per byte). Part 4 switches to BPE, which merges frequent
# byte sequences into single tokens.

class ByteTokenizer:
    """Ultra-simple byte-level tokenizer.
    - encode(str) -> LongTensor [N]
    - decode(Tensor[int]) -> str
    - vocab_size = 256
    """
    def encode(self, s: str) -> torch.Tensor:
        # "hi" -> b"hi" -> [104, 105]
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # errors='ignore': a model can generate byte sequences that aren't valid UTF-8
        # (e.g. half of a multi-byte character) — drop those instead of crashing.
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        # Number of distinct token ids = size of the embedding table and of the output layer.
        return 256