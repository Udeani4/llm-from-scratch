from __future__ import annotations
from typing import List

class LengthCurriculum:
    """6.3 Curriculum: iterate examples from short→long prompts (one pass demo)."""
    # Curriculum learning: show the model easy examples first, harder ones later (like a school
    # syllabus). Prompt length is used as a cheap proxy for difficulty.
    def __init__(self, items: List[tuple[str,str]]):
        # items are (prompt, response) tuples; sort by prompt length, shortest first.
        self.items = sorted(items, key=lambda p: len(p[0]))
        self._i = 0
    # Standard iterator protocol, so `list(LengthCurriculum(...))` yields items in curriculum order.
    def __iter__(self):
        self._i = 0
        return self
    def __next__(self):
        if self._i >= len(self.items):
            raise StopIteration
        it = self.items[self._i]
        self._i += 1
        return it