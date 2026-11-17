from __future__ import annotations
import argparse, torch
from data_prefs import load_preferences
from collator_rm import PairCollator
from model_reward import RewardModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    # (HH-RLHF only has 'train' and 'test' splits, so the 'val' default falls back to the toy pairs)
    p.add_argument('--split', type=str, default='val[:200]')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    items = load_preferences(split=args.split)
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]

    col = PairCollator(block_size=256, bpe_dir=args.bpe_dir)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})

    # Rebuild the reward model with the saved architecture, then load its weights.
    model = RewardModel(vocab_size=cfg.get('vocab_size', col.vocab_size), block_size=cfg.get('block_size', 256),
                        n_layer=cfg.get('n_layer', 4), n_head=cfg.get('n_head', 4), n_embd=cfg.get('n_embd', 256))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()   # eval() disables dropout -> deterministic scores

    # Evaluate accuracy r_pos>r_neg
    # Pairwise accuracy = how often the RM agrees with the human label. 0.5 = coin flip.
    # (Real RMs on HH typically reach ~0.65-0.75 — human preferences are noisy.)
    import math
    B = 16
    correct = 0; total = 0
    for i in range(0, len(triples), B):
        batch = triples[i:i+B]
        pos, neg = col.collate(batch)
        pos, neg = pos.to(device), neg.to(device)
        with torch.no_grad():
            r_pos = model(pos)   # (B,) scores for the human-preferred answers
            r_neg = model(neg)   # (B,) scores for the rejected answers
        correct += (r_pos > r_neg).sum().item()
        total += pos.size(0)
    acc = correct / max(1, total)
    print(f"pairs={total}  accuracy (r_pos>r_neg) = {acc:.3f}")

if __name__ == '__main__':
    main()