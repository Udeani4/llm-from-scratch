from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
from tokenizer_bpe import BPETokenizer

class TextBPEBuffer(Dataset):
    """Memory-mapped-ish single-file dataset: tokenize once → long tensor of ids.
    get(idx) returns a (block_size,) slice; we construct (x,y) with shift inside collate.
    """
    def __init__(self, path: str, tokenizer: BPETokenizer, block_size: int = 256):
        super().__init__()
        self.block_size = block_size
        text = Path(path).read_text(encoding='utf-8')
        # Tokenize the entire corpus once into one long 1-D stream of token ids.
        # With BPE, each id is usually a whole word or word-piece, so the same text yields
        # far fewer tokens than with bytes -> more text fits in the context window.
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    def __len__(self):
        # Every possible start offset is one sample (overlapping sliding windows).
        return max(0, self.ids.numel() - self.block_size - 1)
    def __getitem__(self, i: int):
        # Same next-token shift as Part 2: y is x moved one token to the right.
        x = self.ids[i:i+self.block_size]
        y = self.ids[i+1:i+self.block_size+1]
        return x, y

def make_loader(path: str, tokenizer: BPETokenizer, block_size: int, batch_size: int, shuffle=True) -> DataLoader:
    ds = TextBPEBuffer(path, tokenizer, block_size)
    # DataLoader stacks samples into (B, block_size) batches; drop_last avoids a smaller final batch.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)