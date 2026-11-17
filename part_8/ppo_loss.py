from __future__ import annotations
import torch, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class PPOLossOut:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    total_loss: torch.Tensor


def ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0):
    # All inputs are flat (N,) tensors over the generated (action) tokens in the batch.
    #   new_logp: log-prob of each sampled token under the policy being updated (has gradients)
    #   old_logp: log-prob of the same token under the policy that generated it (fixed)
    #   adv:      advantage — how much better than expected that token turned out
    #   returns:  target for the value head

    # policy
    # ratio = pi_new(token) / pi_old(token). >1 means the update made this token more likely.
    ratio = torch.exp(new_logp - old_logp)  # (N,)
    # Plain policy gradient: raise the probability of tokens with positive advantage, lower negative ones.
    unclipped = ratio * adv
    # PPO's trick: cap the ratio to [1-eps, 1+eps]. Taking the min of clipped/unclipped means once a
    # token's probability has moved more than ~20%, there's no further incentive to push it —
    # this keeps each update small and stable (a "trust region").
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # Negate because optimizers minimise and we want to MAXIMISE the objective.
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    # value (clip optional → here: simple MSE)
    # Train the value head to predict the returns (old_values is unused since value clipping is off).
    value_loss = F.mse_loss(new_values, returns)

    # entropy bonus (we approximate entropy via -new_logp mean; strictly needs full dist)
    # Higher entropy = more exploration / diversity. (Off by default with ent_coef=0.)
    entropy = -new_logp.mean()

    # approx KL for logging
    # How far the policy moved during this update; large values signal the update was too aggressive.
    approx_kl = torch.mean(old_logp - new_logp)

    # Combined objective: improve policy + fit value head - (optionally) reward entropy.
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return PPOLossOut(policy_loss, value_loss, entropy, approx_kl, total)