from __future__ import annotations
import re
from typing import List, Tuple

# Simple automatic metrics for comparing a model's answer to a reference ("gold") answer.
# These are the SQuAD-style QA metrics; they work for short factual answers but say little
# about open-ended responses (that's why Parts 7-9 move to learned reward models).

def _normalize(s: str) -> str:
    # Lowercase, strip punctuation, collapse whitespace, so "Red!" and "red" compare equal.
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, gold: str) -> float:
    # 1.0 only if the normalised strings are identical — strict.
    return float(_normalize(pred) == _normalize(gold))

def token_f1(pred: str, gold: str) -> float:
    # Partial credit based on word overlap (bag of words, order ignored).
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    # Count overlapping words, respecting multiplicity (each gold word can be matched once).
    common = 0
    gp = g.copy()
    for t in p:
        if t in gp:
            gp.remove(t); common += 1
    if common == 0:
        return 0.0
    prec = common / len(p)   # how much of the prediction is correct
    rec  = common / len(g)   # how much of the gold answer was covered
    return 2*prec*rec/(prec+rec)   # harmonic mean