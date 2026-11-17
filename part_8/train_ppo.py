from __future__ import annotations
import argparse, torch
from pathlib import Path

# import torch
# torch.manual_seed(0)

from policy import PolicyWithValue
from rollout import RLHFTokenizer, format_prompt_only, format_example, sample_prompts, gather_logprobs, shift_labels
from rollout import model_logprobs

# Reward model from Part 7
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402

from ppo_loss import ppo_losses

# One PPO iteration =
#   1) ROLLOUT: policy generates responses to a batch of prompts
#   2) SCORE:   reward model gives each full response one scalar reward
#   3) SHAPE:   subtract a per-token KL penalty vs. the frozen SFT reference
#   4) ADVANTAGE: reward minus the value head's prediction (normalised)
#   5) UPDATE:  one gradient step on the PPO clipped objective + value loss
# Three models are in play: policy (trainable), reference (frozen SFT copy), reward model (frozen).


def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt: str, response: str, device) -> float:
    # Re-render prompt+response in the SFT template (the format the RM was trained on) and score it.
    text = format_example(__import__('part_6.formatters', fromlist=['Example']).Example(prompt, response))
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    with torch.no_grad():
        r = reward_model(x)
    return float(r[0].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/ppo-demo')
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)       # prompts per iteration
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--resp_len', type=int, default=64)        # max generated tokens per response
    p.add_argument('--kl_coef', type=float, default=0.01)     # strength of the "stay close to SFT" penalty
    p.add_argument('--gamma', type=float, default=1.0)        # discount factor (unused: no GAE below)
    p.add_argument('--lam', type=float, default=0.95)         # GAE lambda (unused: no GAE below)
    p.add_argument('--lr', type=float, default=1e-5)          # RL uses a much smaller LR than pretraining
    p.add_argument('--bpe_dir', type=str, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # Load SFT policy as initial policy AND reference
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    # Trainable policy, starting from the SFT weights (the value head starts random).
    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])  # initialize LM weights from SFT


    # Frozen copy of the SFT model. The KL penalty against it stops the policy from drifting into
    # degenerate text that happens to fool the reward model (reward hacking).
    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # Reward model
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # small prompt pool
    prompts = sample_prompts(16)

    step = 0
    while step < args.steps:
        # ----- COLLECT ROLLOUT BATCH -----
        # Cycle through the prompt pool batch_size prompts at a time (wrap around at the end).
        batch_prompts = prompts[ (step*args.batch_size) % len(prompts) : ((step+1)*args.batch_size) % len(prompts) ]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size-len(batch_prompts)]
        texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]

        with torch.no_grad():
            out_ids = []
            for i, x in enumerate(in_ids):
                idx = torch.tensor([x], dtype=torch.long, device=device)
                # Sample a response from the CURRENT policy (on-policy RL). Generation stops early at </s>.
                out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=0.2, top_k=3)
                out_ids.append(out[0].tolist())

        # split prompt/response per sample
        data = []
        for i, prompt in enumerate(batch_prompts):
            full = out_ids[i]
            # find boundary: index where prompt ends in the tokenized form
            # Use original prompt tokenization length (clipped by block_size)
            p_ids = in_ids[i][-block_size:]
            boundary = len(p_ids)
            resp_ids = full[boundary:]   # the generated tokens = the "actions" we'll learn from
            # compute rewards via RM on formatted prompt+response text
            resp_text = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_text, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))

        # pad to same length
        # Build batch tensors:
        #   seq     (B,T): prompt+response token ids, right-padded with <pad>=2
        #   mask    (B,T): True on response tokens (the only positions we train on)
        #   rewards (B,T): zero everywhere except the RM score on each sequence's LAST token
        policy_ctx = getattr(policy, "block_size", block_size)
        max_len = min(policy_ctx, max(t[0].numel() for t in data))
        B = len(data)
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)
        rewards = torch.zeros(B, max_len, dtype=torch.float, device=device)

        for i, (ids, boundary, r_scalar) in enumerate(data):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L                 # tokens dropped from the left
            b = max(0, boundary - drop)       # shift boundary after left-trim
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2  # fill remaining positions with <pad> token
            mask[i, b:L] = True
            # Sparse, end-of-episode reward: the RM judges the whole response, so the score is
            # attached to the final token.
            rewards[i, L-1] = r_scalar
            last_idx[i] = L-1


        # logprobs & values for policy and reference
        # model_logprobs returns (B, T-1) for next-token logp; align to seq[:,1:]
        # pol_lp[:, t] = log pi(seq[t+1] | seq[:t+1]) — the probability of the action taken at t+1.
        pol_lp = model_logprobs(policy, seq)
        ref_lp = model_logprobs(ref, seq)
        # values for seq positions (B,T)
        with torch.no_grad():
            logits, values, _ = policy(seq, None)
        values = values[:, :-1]  # align to pol_lp

        # Select only action positions
        # Shift the mask the same way, so we keep entries whose PREDICTED token is a response token.
        act_mask = mask[:,1:]  # since logprobs are for predicting token t from <=t-1
        # Everything below is flattened to (N,) = all response tokens in the batch.
        # "old" = snapshot of the policy that generated the data, frozen for the update.
        old_logp = pol_lp[act_mask].detach()
        ref_logp = ref_lp[act_mask].detach()
        old_values = values[act_mask].detach()

        # KL per action token and shaped rewards
        # Per-token estimate of log(pi/ref); positive where the policy is more confident than SFT was.
        kl = (old_logp - ref_logp)
        # Shaped reward = RM score (last token only) - kl_coef * KL (every token).
        # This is the standard InstructGPT reward: maximise RM score without straying far from SFT.
        shaped_r = rewards[:,1:][act_mask] - args.kl_coef * kl # penalty for drifting

        # Compute advantages/returns with last‑step bootstrap = 0 (episodic per response)
        # Flatten by sequence order inside each sample; we’ll approximate by grouping tokens per sample using last_idx.
        # For tutorial simplicity, treat advantages = shaped_r - old_values (no GAE). Works for end-only reward.
        # Big simplification to be aware of: each token's return is only its OWN shaped reward, so the
        # RM score directly credits just the final token; earlier tokens only see their KL penalty.
        # Proper PPO sums future rewards (with GAE using gamma/lam) so every token shares the credit.
        returns = shaped_r  # target value = immediate shaped reward
        adv = returns - old_values
        # normalize adv
        # Zero-mean / unit-std advantages: standard trick that stabilises the gradient scale.
        adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

        # ----- UPDATE (single pass PPO for demo) -----
        # This step is done multiple times per batch in practice
        # (With one pass, new_logp == old_logp at this point, so ratio == 1 and the clipping never
        #  activates — it's effectively vanilla policy gradient. Clipping matters on later epochs.)
        policy.train()
        # Recompute log-probs WITH gradients this time.
        logits_new, values_new_full, _ = policy(seq, None)
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels = seq[:,1:]
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        new_logp = new_logp_all[act_mask]
        new_values = values_new_full[:, :-1][act_mask]

        from ppo_loss import ppo_losses
        out_loss = ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
                              clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0)
        loss = out_loss.total_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)   # RL gradients are noisy; clip them
        opt.step()
        policy.eval()

        # Diagnostics — the two KLs to watch in RLHF:
        with torch.no_grad():
            # KL(old || new): movement of the updated policy from the snapshot used to collect data
            lp_post = model_logprobs(policy, seq)          # (B, T-1)
            lp_post = lp_post[act_mask]                    # only action positions
            kl_post = (old_logp - lp_post).mean()          # ≈ E[log π_old - log π_new]

            # KL(now || ref): how far the current policy is from the frozen reference
            # (should grow slowly; if it explodes, the policy is drifting/reward-hacking -> raise kl_coef)
            lp_now = lp_post                               # already computed above on the same positions
            kl_ref_now = (lp_now - ref_logp).mean()        # ≈ E[log π_now - log π_ref]

        step += 1
        if step % 10 == 0:
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| value loss {out_loss.value_loss.item():.4f} | KL_move {kl_post.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )


    # Save the full policy (LM + value head); eval_ppo.py loads it.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved PPO policy to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()