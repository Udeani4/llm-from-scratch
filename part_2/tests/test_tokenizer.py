import torch
from tokenizer import ByteTokenizer

def test_roundtrip():
    tok = ByteTokenizer()
    # Includes non-ASCII chars (ä, ö), which become 2 bytes = 2 tokens each in UTF-8.
    s = "Hello, ByteTok! äö"
    ids = tok.encode(s)
    assert ids.dtype == torch.long   # embedding layers require integer (long) ids
    s2 = tok.decode(ids)
    assert len(s2) > 0