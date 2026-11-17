import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    # Gated FFN used by Llama/PaLM. Instead of Linear -> GELU -> Linear, one branch computes
    # values and another computes a smooth 0..1-ish "gate" that decides how much of each value
    # passes through. Empirically gives better quality than plain GELU at the same compute.
    # (Llama shrinks `inner` to ~2/3 * 4 * dim to keep parameter count equal; here it's 4 * dim.)
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False)   # "value" branch
        self.w2 = nn.Linear(dim, inner, bias=False)   # "gate" branch
        self.w3 = nn.Linear(inner, dim, bias=False)   # project back down to model width
        self.act = nn.SiLU()                          # SiLU/Swish: x * sigmoid(x)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        a = self.w1(x)
        b = self.act(self.w2(x))
        # Element-wise gating, then down-projection.
        return self.drop(self.w3(a * b))