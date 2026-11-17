# grpo_loss.py
from __future__ import annotations
import torch
from dataclasses import dataclass

# GRPO (Group Relative Policy Optimization, from DeepSeekMath / DeepSeek-R1) keeps PPO's clipped
# policy objective but DROPS the value head. Instead of a learned critic predicting the expected
# reward, the baseline is the average reward of several answers sampled for the SAME prompt
# (computed in train_grpo.py). That means one less network to train, and no value-estimate errors.
# The KL-to-reference term moves from the reward into the loss.

@dataclass
class PolicyOnlyLossOut:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    kl_ref: torch.Tensor
    total_loss: torch.Tensor


def ppo_policy_only_losses(new_logp, old_logp, adv, clip_ratio=0.2, ent_coef=0.0,
                           kl_coef: float = 0.0, kl_mean: torch.Tensor | None = None):
    """
    PPO-style clipped policy loss, *policy only* (no value head),
    plus a separate KL(π||π_ref) penalty term:  total = L_PPO + kl_coef * KL.
    Inputs are flat over action tokens: new_logp, old_logp, adv: (N_act,).
    kl_mean is a scalar tensor (mean over action tokens).
    """
    device = new_logp.device if new_logp.is_cuda else None
    # Edge case: no response tokens in the batch (e.g. every generation stopped immediately).
    if new_logp.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return PolicyOnlyLossOut(zero, zero, zero, zero, zero)

    # Identical to the PPO policy term in part_8/ppo_loss.py:
    ratio = torch.exp(new_logp - old_logp)  # (N,)          pi_new / pi_old per token
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv   # limit how far one update moves
    policy_loss = -torch.mean(torch.min(unclipped, clipped))   # pessimistic (min) bound, negated to minimise

    # Optional entropy bonus (approximated from the sampled tokens' log-probs).
    entropy = -new_logp.mean() if ent_coef != 0.0 else new_logp.new_tensor(0.0)
    # Logging only: how much the policy moved in this update.
    approx_kl = torch.mean(old_logp - new_logp)

    # KL(pi || pi_ref) computed by the caller WITH gradients, so it directly pulls the policy back
    # toward the SFT reference (PPO in Part 8 instead subtracted it from the reward).
    kl_ref = kl_mean if kl_mean is not None else new_logp.new_tensor(0.0)

    total = policy_loss - ent_coef * entropy + kl_coef * kl_ref # entropy bonus was not used in original GRPO paper
    return PolicyOnlyLossOut(policy_loss, entropy, approx_kl, kl_ref, total)
