from __future__ import annotations
import argparse, torch
from tokenizer import ByteTokenizer
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, default='')
    p.add_argument('--tokens', type=int, default=200)          # how many new tokens to generate
    p.add_argument('--temperature', type=float, default=1.0)   # randomness of sampling
    p.add_argument('--top_k', type=int, default=50)            # only sample from the 50 most likely tokens
    p.add_argument('--top_p', type=float, default=None)        # nucleus sampling threshold (off by default)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = ByteTokenizer()
    # Text -> token ids (N,) -> add a batch dim -> (1, N), since the model expects (B, T).
    prompt_ids = tok.encode(args.prompt).unsqueeze(0).to(device)
    if prompt_ids.numel() == 0:
        # If no prompt provided, seed with newline byte (10)
        # (the model needs at least one token to condition on)
        prompt_ids = torch.tensor([[10]], dtype=torch.long, device=device)


    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt.get('config', None)

    # Rebuild the model with the same architecture it was trained with, then load the weights.
    if config is None:
        # fall back to defaults
        model = GPT(tok.vocab_size, block_size=256).to(device)
        model.load_state_dict(ckpt['model'])
    else:
        model = GPT(**config).to(device)
        model.load_state_dict(ckpt['model'])

    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    # out[0] = prompt + generated tokens for the first (only) sequence; convert ids back to text.
    print(tok.decode(out[0].cpu()))


if __name__ == '__main__':
    main()