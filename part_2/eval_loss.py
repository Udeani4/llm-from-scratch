from __future__ import annotations
import argparse, torch
from dataset import ByteDataset
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)   # number of random val batches to average over
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = ByteDataset(args.data, block_size=args.block_size)
    # map_location lets a checkpoint saved on GPU be loaded on a CPU-only machine.
    ckpt = torch.load(args.ckpt, map_location=device)
    # The checkpoint stores the architecture hyper-parameters so we can rebuild the exact same model;
    # fall back to defaults if they're missing (e.g. model_final.pt doesn't save a config).
    cfg = ckpt.get('config', {
        'vocab_size': 256,
        'block_size': args.block_size,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'dropout': 0.0,
    })
    model = GPT(**cfg).to(device)
    # Copy the trained weights into the freshly built model.
    model.load_state_dict(ckpt['model'])

    # eval() disables dropout; no_grad() skips building the autograd graph (faster, less memory).
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.iters):
            xb, yb = ds.get_batch('val', args.batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    # Mean cross-entropy on held-out data. Lower = better. For bytes, ln(256)≈5.55 is random guessing.
    print(f"val loss: {sum(losses)/len(losses):.4f}")


if __name__ == '__main__':
    main()