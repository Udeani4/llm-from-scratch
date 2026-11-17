from __future__ import annotations
from pathlib import Path
import torch

class ByteDataset:
    """Holds raw bytes of a text file and yields (x,y) blocks for LM.
    - block_size: sequence length (context window)
    - split: fraction for training (rest is val)
    """
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        # Read the file as raw bytes: each byte (0..255) IS a token id with the byte tokenizer,
        # so "tokenizing" the whole corpus is just converting bytes to ints.
        data = Path(path).read_bytes()
        data = torch.tensor(list(data), dtype=torch.long)  # long = int64, required by nn.Embedding
        # First 90% for training, last 10% held out to measure generalisation (val loss).
        n = int(len(data) * split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, which: str, batch_size: int, device: torch.device):
        buf = self.train if which == 'train' else self.val
        assert len(buf) > self.block_size + 1, 'file too small for given block_size'
        # Pick batch_size random start offsets anywhere in the token stream.
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        # x = a window of block_size tokens (the model's input / context).
        x = torch.stack([buf[i:i+self.block_size] for i in ix])
        # y = the same window shifted one token to the right: y[t] is the token that comes after x[t].
        # This shift is the whole "next-token prediction" training objective — one window gives
        # block_size training examples at once (predict token 1 from 0, token 2 from 0..1, ...).
        y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix])
        return x.to(device), y.to(device)