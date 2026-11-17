import argparse, torch
from tokenizer import ByteTokenizer
from model_modern import GPTModern
import time

# Compares generation speed WITH vs WITHOUT the KV cache on an untrained (random-weight) model.
# The output text will be gibberish — the point is timing and checking both paths agree.

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Feature flags to toggle each modern component on/off.
    p.add_argument('--rmsnorm', action='store_true')
    p.add_argument('--rope', action='store_true')
    p.add_argument('--swiglu', action='store_true')
    p.add_argument('--sliding_window', type=int, default=None)
    p.add_argument('--sink', type=int, default=0)
    # Despite the name, this is passed as n_kv_head (number of K/V heads for GQA), not the group size.
    p.add_argument('--group_size', type=int, default=2)
    p.add_argument('--tokens', type=int, default=120)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = ByteTokenizer()
    # 4 query heads sharing args.group_size (=2) K/V heads -> each KV head serves 2 query heads.
    model = GPTModern(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=4, n_embd=128,
                      use_rmsnorm=args.rmsnorm, use_swiglu=args.swiglu, rope=args.rope,
                      max_pos=4096, sliding_window=args.sliding_window, attention_sink=args.sink, n_kv_head=args.group_size).to(device)

    # empty prompt → newline
    prompt = torch.tensor([[10]], dtype=torch.long, device=device)

    with torch.no_grad():
        # temperature=0.0 -> greedy decoding (always pick the most likely token), so both runs are
        # deterministic and should produce the same text if the KV cache is implemented correctly.
        start = time.time()
        out = model.generate(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        print(f"Generated {args.tokens} tokens in {time.time()-start:.2f} sec")

        start = time.time()
        out_nocache = model.generate_nocache(prompt, max_new_tokens=args.tokens, temperature=0.0, top_k=50, top_p=None,
                              sliding_window=args.sliding_window, attention_sink=args.sink)
        print(f"(nocache) Generated {args.tokens} tokens in {time.time()-start:.2f} sec")
    print(tok.decode(out[0].cpu()))
    print(tok.decode(out_nocache[0].cpu()))