from __future__ import annotations
import torch, torch.nn as nn
from gating import TopKGate
from experts import ExpertMLP

class MoE(nn.Module):
    """Mixture‑of‑Experts layer (token‑wise top‑k routing).
    Implementation is single‑GPU friendly (loops over experts for clarity).
    https://arxiv.org/pdf/2101.03961
    """
    # Why MoE? It decouples parameter count from compute. With E experts and k=1, the layer has
    # ~E times the parameters of one FFN (more capacity / knowledge) but each token only pays the
    # compute of ONE FFN. Mixtral, DeepSeek-V3, etc. use this to replace the FFN in each block.
    def __init__(self, dim: int, n_expert: int, k: int = 1, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_expert = n_expert
        self.k = k
        self.gate = TopKGate(dim, n_expert, k=k)   # router
        self.experts = nn.ModuleList([ExpertMLP(dim, mult=mult, swiglu=swiglu, dropout=dropout) for _ in range(n_expert)])

    def forward(self, x: torch.Tensor):
        """x: (B, T, C) → y: (B, T, C), aux_loss
        Steps: flatten tokens → gate → per‑expert forward → scatter back with weights.
        """
        B, T, C = x.shape
        S = B * T
        # Routing is per token, independent of batch/sequence position, so treat all tokens as one list.
        x_flat = x.reshape(S, C)
        idx, w, aux = self.gate(x_flat)  # (S,k), (S,k)

        y = torch.zeros_like(x_flat)     # (S,C)
        for e in range(self.n_expert):
            # tokens where expert e is selected at any of k slots
            for slot in range(self.k):
                # Boolean mask: which tokens chose expert e as their slot-th choice.
                sel = (idx[:, slot] == e)
                if sel.any():
                    x_e = x_flat[sel]                 # gather just those tokens ("dispatch")
                    y_e = self.experts[e](x_e)        # run the expert only on them
                    # Weight by the router's probability and add into place ("combine").
                    # slot:slot+1 keeps shape (N,1) so it broadcasts over C.
                    y[sel] += w[sel, slot:slot+1] * y_e
        y = y.view(B, T, C)
        # aux must be added (times a small coefficient) to the training loss by the caller.
        return y, aux