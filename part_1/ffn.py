import torch.nn as nn

class FeedForward(nn.Module):
    """1.5 FFN with expansion factor `mult`.

    Dimensions:
      input:     (B, T, d_model)
      inner:     (B, T, mult*d_model)
      output:    (B, T, d_model)

    `mult*d_model` means the hidden width is `mult` times larger than `d_model`.
    Typical values: mult=4 for GELU FFN in GPT-style blocks.
    """
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # A 2-layer MLP applied to every token independently. Most of a transformer's parameters
        # (and much of its stored "knowledge") live in these layers.
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),   # expand to a wider hidden space
            nn.GELU(),                            # smooth ReLU-like non-linearity used by GPT/BERT
            nn.Linear(mult * d_model, d_model),   # project back so the residual add shapes match
            nn.Dropout(dropout),                  # regularisation (randomly zeroes activations in training)
        )

    def forward(self, x):
        return self.net(x)