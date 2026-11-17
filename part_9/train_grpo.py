# train_grpo.py
from __future__ import annotations
import argparse, torch
from pathlib import Path

from policy import PolicyWithValue  # we will ignore the value head
from rollout import RLHFTokenizer, format_prompt_only, sample_prompts, model_logprobs

# Reward model from Part 7
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402

from grpo_loss import ppo_policy_only_losses

# GRPO training loop. Differences from PPO (part_8/train_ppo.py):
#   - For each prompt, sample a GROUP of G answers.
#   - Advantage of an answer = its reward - the group's average reward. "Was this answer better or
#     worse than my other attempts at the same question?" No value head / critic needed.
#   - That one advantage is shared by every token of the answer (trajectory-level credit).
#   - KL to the reference model is a loss term, not part of the reward.


@torch.no_grad()
def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt_text: str, response_ids: list[int], device) -> float:
    # Build full formatted text (as in your PPO)
    # Score "template(prompt, response)" with the frozen reward model -> one scalar per answer.
    from part_6.formatters import Example, format_example
    resp_text = tok.decode(response_ids)
    text = format_example(Example(prompt_text, resp_text))
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    r = reward_model(x)
    return float(r[0].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/grpo-demo')
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch_prompts', type=int, default=32, help='number of distinct prompts per step (before grouping)')
    p.add_argument('--group_size', type=int, default=4, help='completions per prompt')
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--resp_len', type=int, default=64)
    p.add_argument('--kl_coef', type=float, default=0.01)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bpe_dir', type=str, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # Load SFT policy (and a frozen reference)
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    # Trainable policy initialised from SFT. (PolicyWithValue is reused for convenience; its value
    # head is never trained here.)
    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])
    policy.eval()

    # Frozen SFT reference for the KL penalty.
    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # Reward model
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size),
                     block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4),
                     n_head=rckpt['config'].get('n_head', 4),
                     n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # small prompt pool (reuse your helper)
    prompts_pool = sample_prompts(16)

    step = 0
    pool_idx = 0
    G = args.group_size

    while step < args.steps:
        # ----- SELECT PROMPTS -----
        # Choose P prompts, each will yield G completions → B = P*G trajectories
        P = max(1, args.batch_prompts)
        if pool_idx + P > len(prompts_pool):
            pool_idx = 0
        batch_prompts = prompts_pool[pool_idx: pool_idx + P]
        pool_idx += P

        # Tokenize prompt-only texts
        prompt_texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        prompt_in_ids = [tok.encode(t) for t in prompt_texts]

        # ----- GENERATE G COMPLETIONS PER PROMPT -----
        # We will collect all trajectories flat, but track their group/prompt ids.
        seq_list = []        # list[Tensor of token ids]
        boundary_list = []   # index where response starts in the (possibly clipped) sequence
        prompt_id_of = []    # which prompt this trajectory belongs to (0..P-1)
        raw_rewards = []     # scalar reward per trajectory (before KL shaping)
        last_idx_list = []   # for padding bookkeeping

        with torch.no_grad():
            for pid, p_ids in enumerate(prompt_in_ids):
                for g in range(G):
                    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
                    # High temperature (2) on purpose: the G answers must DIFFER for the group comparison
                    # to carry any signal. If all G answers were identical, every advantage would be 0.
                    out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=2, top_k=3)
                    full_ids = out[0].tolist()

                    # split prompt/response
                    boundary = len(p_ids[-block_size:])  # prompt length clipped to context
                    resp_ids = full_ids[boundary:]
                    r_scalar = compute_reward(rm, tok, batch_prompts[pid], resp_ids, device)

                    seq_list.append(torch.tensor(full_ids, dtype=torch.long))
                    boundary_list.append(boundary)
                    prompt_id_of.append(pid)
                    raw_rewards.append(r_scalar)

        # ----- PAD TO BATCH -----
        # Same batching as PPO: seq (B,T) right-padded with <pad>=2, mask True on response tokens.
        B = len(seq_list)  # B = P*G
        policy_ctx = getattr(policy, "block_size", block_size)
        max_len = min(policy_ctx, max(s.numel() for s in seq_list))
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)

        # keep a per-traj “action positions” mask and response-only boundary
        for i, (ids, bnd) in enumerate(zip(seq_list, boundary_list)):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L
            b = max(0, bnd - drop)  # shifted boundary after left-trim
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2  # pad token
            # actions are predicting token t from <=t-1 → positions [1..L-1]
            # but we only care about response tokens: mask [b..L-1] → actions [b+1..L-1]
            mask[i, b:L] = True
            last_idx[i] = L - 1

        # ----- LOGPROBS & KL VS REF (token-level) -----
        # model_logprobs returns log p(x[t] | x[:t-1]) for t=1..T-1 over labels=seq[:,1:]
        with torch.no_grad():
            pol_lp_full = model_logprobs(policy, seq)  # (B, T-1)
            ref_lp_full = model_logprobs(ref, seq)     # (B, T-1)

        # action positions (predict positions [1..T-1]); we want only response tokens:
        act_mask = mask[:, 1:]  # align to (B, T-1)
        # Flattened over all response tokens of all B trajectories -> (N_act,)
        old_logp = pol_lp_full[act_mask].detach()
        ref_logp = ref_lp_full[act_mask].detach()

        # per-token KL on action tokens
        # (computed for reference/logging; the KL actually used in the loss is recomputed below with gradients)
        kl_tok = (old_logp - ref_logp)  # (N_act,)

        # ----- SHAPED TRAJECTORY REWARD & GROUP BASELINE -----
        # For GRPO, advantage is trajectory-level and broadcast to its tokens.
        # We include KL shaping at trajectory level using mean token KL per trajectory.
        # First, compute mean KL per trajectory on its action tokens.
        # Build an index map from flat action tokens back to traj ids.
        # We can reconstruct counts by iterating rows.
        # traj_id_for_token[j] = which trajectory the j-th flattened action token belongs to.
        # Boolean-mask flattening goes row by row, so trajectory i contributes n_i consecutive entries.
        traj_id_for_token = []
        counts = torch.zeros(B, dtype=torch.long, device=device)
        offset = 0
        for i in range(B):
            mrow = act_mask[i]
            n_i = int(mrow.sum().item())
            if n_i > 0:
                traj_id_for_token.extend([i] * n_i)
            counts[i] = n_i
            offset += n_i
        traj_id_for_token = torch.tensor(traj_id_for_token, dtype=torch.long, device=device)
        raw_rewards_t = torch.tensor(raw_rewards, dtype=torch.float, device=device)

        # Compute per-prompt group mean of shaped rewards
        # (These are the raw RM rewards; KL is handled in the loss instead.)
        # This group mean is GRPO's baseline — it replaces PPO's value head.
        group_mean = torch.zeros(B, dtype=torch.float, device=device)
        for pid in range(P):
            idxs = [i for i in range(B) if prompt_id_of[i] == pid]
            if not idxs:
                continue
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
            mean_val = raw_rewards_t[idxs_t].mean()
            group_mean[idxs_t] = mean_val

        # Advantage per trajectory, broadcast to its action tokens
        # > 0: better than this prompt's average answer -> make all its tokens more likely; < 0: less likely.
        # (The GRPO paper also divides by the group's std; here normalisation is done batch-wide below.)
        traj_adv = raw_rewards_t - group_mean  # (B,)

        # Build a flat tensor of advantages aligned with old_logp/new_logp on action tokens
        if kl_tok.numel() > 0:
            adv_flat = traj_adv[traj_id_for_token]
        else:
            adv_flat = torch.zeros(0, dtype=torch.float, device=device)

        # Normalize advantages (optional but usually helpful)
        if adv_flat.numel() > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std().clamp_min(1e-6))

        # ----- UPDATE (policy-only PPO clipped objective) -----
        policy.train()
        # Recompute token log-probs WITH gradients.
        logits_new, _, _ = policy(seq, None)  # ignore value head
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels = seq[:, 1:]
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        new_logp = new_logp_all[act_mask]

        # Mean KL over action tokens
        # Has gradients through new_logp, so minimising it pulls the policy back toward the reference.
        kl_now_ref_mean = (new_logp - ref_logp).mean() if new_logp.numel() > 0 else torch.tensor(0.0, device=device)

        out_loss = ppo_policy_only_losses(
            new_logp=new_logp,
            old_logp=old_logp,
            adv=adv_flat,
            clip_ratio=0.2,
            ent_coef=0.0,  # set >0 if you want entropy bonus from -new_logp mean
            kl_coef=args.kl_coef,
            kl_mean=kl_now_ref_mean,
        )
        loss = out_loss.total_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()

        # Some quick diagnostics (movement vs old, and now vs ref)
        with torch.no_grad():
            lp_post = model_logprobs(policy, seq)[act_mask]
            # How much this single update changed the policy on the sampled tokens.
            kl_move = (old_logp - lp_post).mean() if lp_post.numel() > 0 else torch.tensor(0.0, device=device)
            # KL(now || ref)
            # Total drift from the SFT model so far.
            kl_ref_now = (lp_post - ref_logp).mean() if lp_post.numel() > 0 else torch.tensor(0.0, device=device)

        step += 1
        if step % 10 == 0:
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| KL_move {kl_move.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )

    # Saved in the same format as Part 8, so eval_ppo.py can score it.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved GRPO policy to {args.out}/model_last.pt")


if __name__ == '__main__':
    main()