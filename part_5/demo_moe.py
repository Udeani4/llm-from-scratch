import argparse, torch
from moe import MoE

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=64)     # total tokens (split across a batch of 2)
    p.add_argument('--hidden', type=int, default=128)    # model width C
    p.add_argument('--experts', type=int, default=4)     # number of expert FFNs
    p.add_argument('--top_k', type=int, default=1)       # experts used per token
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    # Random hidden states standing in for the output of an attention layer.
    x = torch.randn(2, args.tokens//2, args.hidden, device=device)  # (B=2,T=tokens/2,C)

    moe = MoE(dim=args.hidden, n_expert=args.experts, k=args.top_k).to(device)
    with torch.no_grad():
        y, aux = moe(x)

    # simple routing histogram
    # Re-run just the router to see which expert each token's top-1 choice was.
    from gating import TopKGate
    gate = moe.gate
    idx, w, _ = gate(x.view(-1, args.hidden))     # flatten (B,T,C) -> (B*T, C) tokens
    # Count tokens per expert. Uneven counts = load imbalance (some experts overworked, some idle).
    hist = torch.bincount(idx[:,0], minlength=args.experts)
    # aux ~= 1.0 means perfectly balanced; larger means routing is skewed toward a few experts.
    print(f"Output shape: {tuple(y.shape)} | aux={float(aux):.4f}")
    print("Primary expert load (counts):", hist.tolist())