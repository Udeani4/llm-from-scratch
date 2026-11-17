from __future__ import annotations
import torch, torch.nn.functional as F

# Reward scores have no absolute meaning; we only have "A is better than B" labels.
# So these losses train on the DIFFERENCE r_pos - r_neg, pushing chosen scores above rejected ones.

def bradley_terry_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    """-log sigma(r_pos - r_neg) = softplus(-(r_pos - r_neg))
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html"""
    # Bradley-Terry model: P(chosen beats rejected) = sigmoid(r_pos - r_neg).
    # Minimising -log of that probability is just logistic regression on the score gap.
    # This is the loss used for InstructGPT / Llama-2 reward models.
    diff = r_pos - r_neg
    # softplus(-d) = log(1 + e^-d): mathematically equal to -log(sigmoid(d)), but numerically stable.
    return F.softplus(-diff).mean()


def margin_ranking_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html"""
    # Hinge alternative: loss = max(0, margin - (r_pos - r_neg)).
    # Zero once chosen beats rejected by at least `margin`; unlike BT it stops pushing after that.
    y = torch.ones_like(r_pos)   # y=1 means "the first argument should rank higher"
    return F.margin_ranking_loss(r_pos, r_neg, y, margin=margin)