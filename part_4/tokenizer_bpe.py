from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

# BPE (Byte-Pair Encoding) — the tokenizer family used by GPT-2/3/4, Llama, etc.
# Training: start from the 256 single bytes, then repeatedly find the most frequent ADJACENT PAIR
# of tokens in the corpus and merge it into a new token ("t"+"h" -> "th", "th"+"e" -> "the", ...),
# until the vocabulary reaches vocab_size.
# Result: common words become single tokens, rare words split into a few pieces, and any text
# can still be encoded (it can always fall back to raw bytes). Sequences are ~3-4x shorter than bytes.

try:
    # HuggingFace's fast Rust implementation.
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
except Exception:
    ByteLevelBPETokenizer = None

class BPETokenizer:
    """Minimal BPE wrapper (HuggingFace tokenizers).
    Trains on a text file or a folder of .txt files. Saves merges/vocab to out_dir.
    """
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] | None = None):
        if ByteLevelBPETokenizer is None:
            raise ImportError("Please `pip install tokenizers` for BPETokenizer.")
        self.vocab_size = vocab_size   # target vocab size (the actual one can be smaller on tiny corpora)
        # Special tokens get reserved ids and are never split. Their ids follow list order:
        # <s>=0 (begin), </s>=1 (end-of-sequence), <pad>=2, <unk>=3, <mask>=4.
        # (That's why Part 6's generate() uses eos_id=1 to stop.)
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, data_path: Union[str, Path]):
        # Accept a single file or a directory of .txt files.
        files: List[str] = []
        p = Path(data_path)
        if p.is_dir():
            files = [str(fp) for fp in p.glob("**/*.txt")]
        else:
            files = [str(p)]
        # "Byte-level": operates on UTF-8 bytes, so there are never unknown characters.
        tok = ByteLevelBPETokenizer()
        # min_frequency=2: only merge pairs that occur at least twice.
        tok.train(files=files, vocab_size=self.vocab_size, min_frequency=2, special_tokens=self.special_tokens)
        self._tok = tok

    def save(self, out_dir: Union[str, Path]):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        assert self._tok is not None, "Train or load before save()."
        # vocab.json (token string -> id) + merges.txt (ordered merge rules)
        self._tok.save_model(str(out))
        # tokenizer.json: single-file full serialisation (this is what load() actually uses)
        self._tok.save(str(out / "tokenizer.json"))
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        (out/"bpe_meta.json").write_text(json.dumps(meta))

    def load(self, dir_path: Union[str, Path]):
        dirp = Path(dir_path)
        # Prefer explicit filenames; fall back to glob if needed.
        vocab = dirp / "vocab.json"
        merges = dirp / "merges.txt"
        tokenizer = dirp / "tokenizer.json"
        if not vocab.exists() or not merges.exists():
            # Fallback for custom basenames
            vs = list(dirp.glob("*.json"))
            ms = list(dirp.glob("*.txt"))
            if not vs or not ms:
                raise FileNotFoundError(f"Could not find vocab.json/merges.txt in {dirp}")
            vocab = vs[0]
            merges = ms[0]
        # tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        tok = Tokenizer.from_file(str(tokenizer))
        self._tok = tok
        meta_file = dirp / "bpe_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            # Note: this restores the REQUESTED vocab_size, used to size the model's embedding table.
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)


    def encode(self, text: str):
        # text -> list[int]  e.g. "hello world" -> [31373, 995]-style ids
        ids = self._tok.encode(text).ids
        return ids

    def decode(self, ids):
        # list[int] -> text
        return self._tok.decode(ids)