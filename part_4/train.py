from __future__ import annotations
import argparse, time, signal
from pathlib import Path
import sys

import torch
import torch.nn as nn

# so we can import Part 3 model
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1] / 'part_3'))
from model_modern import GPTModern

from tokenizer_bpe import BPETokenizer
from dataset_bpe import make_loader
from lr_scheduler import WarmupCosineLR
from amp_accum import AmpGrad
from checkpointing import (
    load_checkpoint,
    _log_hparams_tb,
    _maybe_log_graph_tb,
    _is_tb,
    _log_model_stats,
    _maybe_log_attention,
    _log_samples_tb,
    _log_runtime,
    atomic_save_all,
)
from logger import init_logger

# Part 4 training loop = Part 2's loop + the "scaling up" machinery:
# BPE tokenizer, warmup+cosine LR, mixed precision, gradient accumulation, resumable checkpoints, logging.


def run_cfg_from_args(args, vocab_size: int) -> dict:
    # Constructor kwargs for GPTModern; saved into every checkpoint so the model can be rebuilt.
    return dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_rmsnorm=True,
        use_swiglu=True,
        rope=True,
        max_pos=4096,
        sliding_window=None,
        attention_sink=0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out', type=str, default='runs/part4')

    # tokenizer / model dims
    p.add_argument('--bpe', action='store_true', help='train and use a BPE tokenizer (recommended)')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_embd', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)

    # train
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps', type=int, default=300, help='max optimizer steps for this run')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=20)
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--grad_accum_steps', type=int, default=4)

    # misc
    p.add_argument('--log', choices=['wandb', 'tensorboard', 'none'], default='tensorboard')
    p.add_argument('--save_every', type=int, default=50, help='save checkpoint every N optimizer steps')
    p.add_argument('--keep_last_k', type=int, default=2, help='keep last K step checkpoints (plus model_last.pt)')
    args = p.parse_args()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # output dir and (possible) checkpoint
    # If out_dir already has model_last.pt, this run RESUMES it instead of starting fresh.
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model_last.pt"
    have_ckpt = ckpt_path.exists()

    # ---- load checkpoint meta if present ----
    ckpt = None
    saved_tok_dir = None
    if have_ckpt:
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "config" not in ckpt:
            raise RuntimeError(
                "Checkpoint is missing 'config'."
                "Please re-save a checkpoint that includes the model config."
            )
        tok_file = ckpt_path.with_name("tokenizer_dir.txt")
        saved_tok_dir = tok_file.read_text().strip() if tok_file.exists() else None

    # ---- tokenizer ----
    tok = None
    tok_dir = None
    if have_ckpt:
        # On resume we MUST reuse the original tokenizer: retraining could assign different ids,
        # and the learned embeddings would then point at the wrong tokens.
        if not saved_tok_dir:
            raise RuntimeError(
                "Checkpoint was found but tokenizer_dir.txt is missing. "
                "Resume requires the original tokenizer."
            )
        tok = BPETokenizer(); tok.load(saved_tok_dir)
        tok_dir = saved_tok_dir
        vocab_size = tok.vocab_size
        print(f"[resume] Loaded tokenizer from {tok_dir} (vocab={vocab_size})")
    else:
        if args.bpe:
            # Fresh run: learn a BPE vocabulary from the training data itself, and save it.
            tok = BPETokenizer(vocab_size=args.vocab_size)
            tok.train(args.data)
            tok_dir = str(out_dir / 'tokenizer'); Path(tok_dir).mkdir(parents=True, exist_ok=True)
            tok.save(tok_dir)
            vocab_size = tok.vocab_size
            print(f"[init] Trained tokenizer to {tok_dir} (vocab={vocab_size})")
        else:
            tok = None
            vocab_size = 256  # byte-level fallback (not recommended for Part 4)
            # (note: make_loader below needs a tokenizer, so in practice --bpe is required)

    # ---- dataset ----
    train_loader = make_loader(args.data, tok, args.block_size, args.batch_size, shuffle=True)

    # ---- build model config ----
    if have_ckpt:
        cfg_build = ckpt["config"]
        # Embedding/output layer sizes depend on vocab, so a vocab change would break the weights.
        if cfg_build.get("vocab_size") != vocab_size:
            raise RuntimeError(
                f"Tokenizer vocab ({vocab_size}) != checkpoint config vocab ({cfg_build.get('vocab_size')}). "
                "This deterministic script forbids vocab changes on resume."
            )
    else:
        cfg_build = run_cfg_from_args(args, vocab_size)

    # ---- init model/opt/sched/amp ----
    model = GPTModern(**cfg_build).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Number of OPTIMIZER steps (not micro-batches) the LR schedule spans.
    # (len(train_loader) counts micro-batches, so with grad accumulation this overestimates the
    #  steps one epoch can provide; with --steps set, args.steps is usually the binding limit.)
    total_steps = min(args.steps, args.epochs * len(train_loader))
    # Warmup is capped at 10% of training so short runs aren't all warmup.
    warmup = min(args.warmup_steps, max(total_steps // 10, 1))
    sched = WarmupCosineLR(optim, warmup_steps=warmup, total_steps=total_steps, base_lr=args.lr)

    amp = AmpGrad(optim, accum=args.grad_accum_steps, amp=args.mixed_precision)

    # ---- strict resume ----
    step = 0
    if have_ckpt:
        step = load_checkpoint(model, str(ckpt_path), optimizer=optim, scheduler=sched, amp=amp, strict=True)
        print(f"[resume] Loaded checkpoint at step {step}")

    # ---- logging ----
    logger = init_logger(args.log, out_dir=str(out_dir))
    _log_hparams_tb(logger, args, total_steps)
    if _is_tb(logger):
        try:
            ex_x, ex_y = next(iter(train_loader))
            _maybe_log_graph_tb(logger, model, ex_x.to(device), ex_y.to(device))
        except Exception:
            pass

    # ---- graceful save on SIGINT/SIGTERM ----
    # Ctrl+C (or a cluster job being pre-empted) just sets a flag; the loop then saves a checkpoint
    # at a clean point and exits, so no progress is lost.
    save_requested = {"flag": False}
    def _on_term(sig, frame): save_requested["flag"] = True
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT,  _on_term)

    # ---- train loop ----
    model.train()
    while step < args.steps:                 # loop over epochs until the step budget is used up
        for xb, yb in train_loader:          # each xb/yb is one MICRO-batch
            if step >= args.steps: break
            if save_requested["flag"]:
                atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                print(f"[signal] Saved checkpoint at step {step} to {out_dir}. Exiting.")
                return

            it_t0 = time.time()
            xb, yb = xb.to(device), yb.to(device)
            # Forward in fp16 where safe (autocast keeps sensitive ops like softmax/loss in fp32).
            with torch.cuda.amp.autocast(enabled=amp.amp):
                logits, loss, _ = model(xb, yb)
            # Accumulate this micro-batch's (scaled) gradients.
            amp.backward(loss)

            # Every grad_accum_steps micro-batches: one real parameter update.
            # (No gradient clipping here, unlike Part 2.)
            if amp.should_step():
                amp.step(); amp.zero_grad()
                lr = sched.step()            # advance the LR schedule once per optimizer step
                step += 1

                # periodic checkpoint
                if step % args.save_every == 0:
                    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                    if _is_tb(logger):
                        logger.text("meta/checkpoint", f"Saved at step {step}", step)

                # logging
                if step % 50 == 0:
                    # loss shown is the last micro-batch's (unscaled) loss.
                    logger.log(step=step, loss=float(loss.item()), lr=float(lr))
                    _log_runtime(logger, step, it_t0, xb, device)
                    _log_model_stats(logger, model, step, do_hists=False)
                    _maybe_log_attention(logger, model, xb, step, every=100)
                    _log_samples_tb(logger, model, tok, xb, device, step, max_new_tokens=64)

    # ---- final save ----
    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
    print(f"Saved checkpoint to {out_dir}/model_last.pt")


if __name__ == '__main__':
    main()
