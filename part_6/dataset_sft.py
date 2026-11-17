from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import traceback

try:
    # HuggingFace `datasets` library: downloads and caches public datasets from the HF Hub.
    from datasets import load_dataset
except Exception:
    print("Couldn't import `datasets`. Will use fallback data only.")
    load_dataset = None

from formatters import Example

@dataclass
class SFTItem:
    # One instruction-tuning example: what the user asks, and the ideal answer.
    prompt: str
    response: str


def load_tiny_hf(split: str = "train[:200]", sample_dataset: bool = False) -> List[SFTItem]:
    """Try to load a tiny instruction dataset from HF; fall back to a baked-in list.
    We use `tatsu-lab/alpaca` as a familiar schema (instruction, input, output) and keep only a slice.
    """
    # Alpaca = 52k instruction/answer pairs (generated with GPT-3), a classic early SFT dataset.
    # split="train[:200]" is HF slicing syntax: only the first 200 rows.
    items: List[SFTItem] = []
    if load_dataset is not None and not sample_dataset:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split=split)
            for row in ds:
                instr = row.get("instruction", "").strip()
                inp = row.get("input", "").strip()      # optional extra context (e.g. text to summarise)
                out = row.get("output", "").strip()
                # Fold the optional input into the prompt so we have a simple (prompt, response) pair.
                if inp:
                    instr = instr + "\n" + inp
                if instr and out:
                    items.append(SFTItem(prompt=instr, response=out))
        except Exception:
            pass   # e.g. no internet -> fall through to the built-in examples
    if not items:
        # fallback tiny list
        seeds = [
            ("First prime number", "2"),
            ("What are the three primary colors?", "red"),
            ("Device name which points to direction?", "compass"),
        ]
        items = [SFTItem(prompt=p, response=r) for p,r in seeds]
    return items