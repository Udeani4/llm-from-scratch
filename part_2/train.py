from __future__ import annotations
import argparse, time
import torch
from tokenizer import ByteTokenizer
from dataset import ByteDataset
from model_gpt import GPT


def estimate_loss(model: GPT, ds: ByteDataset, args) -> dict:
    # Average loss over several random batches for both splits. A single batch is too noisy.
    # If train loss keeps falling but val loss rises, the model is overfitting (memorising).
    model.eval()   # disable dropout for evaluation
    out = {}
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = []
            for _ in range(args.eval_iters):
                xb, yb = ds.get_batch(split, args.batch_size, args.device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    model.train()  # back to training mode
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='runs/min-gpt')
    # --- model size / context ---
    p.add_argument('--block_size', type=int, default=256)   # context length in tokens
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_layer', type=int, default=4)        # number of transformer blocks
    p.add_argument('--n_head', type=int, default=4)         # attention heads per block
    p.add_argument('--n_embd', type=int, default=256)       # hidden size (d_model)
    p.add_argument('--dropout', type=float, default=0.0)
    # --- optimisation ---
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    # --- evaluation / sampling during training ---
    p.add_argument('--eval_interval', type=int, default=200)
    p.add_argument('--eval_iters', type=int, default=50)
    p.add_argument('--sample_every', type=int, default=200)
    p.add_argument('--sample_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=None)
    # --- performance ---
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--amp', action='store_true')
    args = p.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = ByteTokenizer()
    ds = ByteDataset(args.data, block_size=args.block_size)
    model = GPT(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(args.device)

    # torch.compile (PyTorch 2.x) JIT-compiles the model into fused kernels for speed.
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # AdamW = Adam with decoupled weight decay; the standard optimiser for transformers.
    # betas=(0.9, 0.95) is the common LLM setting (beta2 lower than default 0.999 for stability).
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # GradScaler is for mixed precision (fp16): it multiplies the loss by a large factor before
    # backward so tiny gradients don't underflow to 0 in fp16, then un-scales before the update.
    # When disabled it's a no-op pass-through, so the code below works either way.
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.device.type == 'cuda'))

    best_val = float('inf')
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        # 1) sample a random batch of (input, next-token target) windows
        xb, yb = ds.get_batch('train', args.batch_size, args.device)
        # 2) forward pass; autocast runs matmuls in fp16 when AMP is on (faster, less memory)
        with torch.cuda.amp.autocast(enabled=(args.amp and args.device.type == 'cuda')):
            _, loss = model(xb, yb)
        # 3) clear old gradients (set_to_none is slightly faster than zero-filling)
        opt.zero_grad(set_to_none=True)
        # 4) backprop: compute d(loss)/d(every parameter)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            # Gradients must be un-scaled before clipping so the threshold applies to their true size.
            scaler.unscale_(opt)
            # Gradient clipping: if the overall gradient norm exceeds grad_clip, scale it down.
            # Protects against occasional huge gradient spikes that would wreck training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # 5) apply the parameter update (scaler skips the step if it detected inf/NaN gradients)
        scaler.step(opt)
        scaler.update()   # adjust the loss-scale factor for next step

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {(time.time()-t0):.1f}s")
            t0 = time.time()

        if step % args.eval_interval == 0:
            losses = estimate_loss(model, ds, args)
            print(f"eval | train {losses['train']:.4f} | val {losses['val']:.4f}")
            # Keep only the checkpoint with the best validation loss (a simple form of early stopping).
            if losses['val'] < best_val:
                best_val = losses['val']
                ckpt_path = f"{args.out_dir}/model_best.pt"
                import os; os.makedirs(args.out_dir, exist_ok=True)
                # Save weights (state_dict) + the hyper-parameters needed to rebuild the architecture.
                torch.save({'model': model.state_dict(), 'config': {
                    'vocab_size': tok.vocab_size,
                    'block_size': args.block_size,
                    'n_layer': args.n_layer,
                    'n_head': args.n_head,
                    'n_embd': args.n_embd,
                    'dropout': args.dropout,
                }}, ckpt_path)
                print(f"saved checkpoint: {ckpt_path}")

        if args.sample_every > 0 and step % args.sample_every == 0:
            # Qualitative check: seed with a random chunk of real training text and let the model continue it.
            start = torch.randint(low=0, high=len(ds.train) - args.block_size - 1, size=(1,)).item()
            seed = ds.train[start:start + args.block_size].unsqueeze(0).to(args.device)
            out = model.generate(seed, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            txt = tok.decode(out[0].cpu())
            print("\n================ SAMPLE ================\n" + txt[-(args.block_size + args.sample_tokens):] + "\n=======================================\n")
            # Note: generate() calls self.eval(); the next estimate_loss() call switches back to train mode.

    # final save
    import os; os.makedirs(args.out_dir, exist_ok=True)
    torch.save({'model': model.state_dict()}, f"{args.out_dir}/model_final.pt")


if __name__ == '__main__':
    main()