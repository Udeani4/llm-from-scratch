import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    # Simpler/cheaper alternative to LayerNorm used by Llama, Mistral, etc.
    # LayerNorm: subtract mean, divide by std, then scale + shift.
    # RMSNorm:   skip the mean-subtraction and the shift; only rescale by the root-mean-square.
    # Works just as well in practice, with fewer operations.
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps   # avoids division by zero for an all-zero vector
        # Learned per-feature gain `g`, initialised to 1 (so it starts as pure normalisation).
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS over the feature dim of each token, shape (..., 1) so it broadcasts back.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight