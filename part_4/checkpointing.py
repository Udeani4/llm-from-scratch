from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
# Make part_3 importable (for SwiGLU in _extract_config_from_model).
sys.path.append(str(Path(__file__).resolve().parents[1]/'part_3'))
import time
import torch
import shutil
import torch.nn as nn

DEF_NAME = "model_last.pt"

# This file has two halves:
#   1) TensorBoard logging helpers — diagnostics you watch while training.
#   2) Checkpoint save/load — so a long training run can be stopped and resumed exactly.

# ----------------------------- TB-only helpers (safe no-ops otherwise) ----------------------------- #
def _is_tb(logger) -> bool:
    # TBLogger stores its SummaryWriter in `.w`; other loggers don't have one.
    return getattr(logger, "w", None) is not None


# checkpointing._log_hparams_tb
def _log_hparams_tb(logger, args, total_steps):
    # Record the run's hyper-parameters once, so runs can be compared side by side in TensorBoard.
    if not _is_tb(logger): return
    try:
        h = dict(
            vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
            n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, lr=args.lr,
            warmup_steps=args.warmup_steps, batch_size=args.batch_size, grad_accum=args.grad_accum_steps,
            mixed_precision=args.mixed_precision, steps=args.steps, epochs=args.epochs,
        )
        logger.hparams(h, {"meta/total_steps": float(total_steps)})
    except Exception:
        pass

def _maybe_log_graph_tb(logger, model, xb, yb):
    # Draw the model's computation graph in TensorBoard's "Graphs" tab.
    if not hasattr(logger, "graph"):
        return
    try:
        # The tracer needs forward() to return a tensor, but GPTModern returns (logits, loss, caches);
        # this tiny wrapper returns only the first tensor.
        class _TensorOnly(nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m.eval()
            def forward(self, x, y=None):
                out = self.m(x, y) if y is not None else self.m(x)
                if isinstance(out, (list, tuple)):
                    for o in out:
                        if torch.is_tensor(o):
                            return o
                    return out[0]
                return out
        wrapped = _TensorOnly(model).to(xb.device)
        logger.graph(wrapped, (xb, yb))
        # Note: m.eval() above also switches the real model to eval mode (it's the same object).
    except Exception:
        pass

def _log_model_stats(logger, model, step: int, do_hists: bool = False):
    # Global L2 norms of all weights and all gradients — key training-health signals:
    # exploding grad norm -> instability; grad norm ~0 -> learning has stalled.
    if not _is_tb(logger): return
    try:
        params = [p for p in model.parameters() if p.requires_grad]
        # Norm of the vector of per-tensor norms = norm over all parameters concatenated.
        total_param_norm = torch.norm(torch.stack([p.detach().norm(2) for p in params]), 2).item()
        # (In train.py this runs right after zero_grad(set_to_none=True), so grads are None and this logs NaN.)
        grads = [p.grad for p in params if p.grad is not None]
        total_grad_norm = float('nan')
        if grads:
            total_grad_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()
        logger.log(step=step, **{
            "train/param_global_l2": total_param_norm,
            "train/grad_global_l2": total_grad_norm,
        })
        if do_hists:
            # Per-layer distributions of weights and grads (expensive, so off by default).
            for name, p in model.named_parameters():
                logger.hist(f"params/{name}", p, step)
                if p.grad is not None:
                    logger.hist(f"grads/{name}", p.grad, step)
    except Exception:
        pass

def _maybe_log_attention(logger, model, xb, step: int, every: int = 100):
    """
    Logs Q/K/V histograms for each Transformer block using the current minibatch xb.
    No model edits. No hooks. Runs a light no-grad recomputation of the pre-attn path.
    - Takes first batch and first head only to keep logs tiny.
    - Uses pre-RoPE values (simpler & stable for histograms).
    """
    # Why watch Q/K/V? If their magnitudes blow up, attention logits get huge and softmax saturates
    # (attention "entropy collapse") — a common cause of loss spikes in LLM training.
    if not _is_tb(logger) or step == 0 or (step % every):
        return
    try:
        import torch
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            # Recreate inputs seen by blocks
            x = model.tok_emb(xb)           # (B,T,C)
            x = model.drop(x)

            B, T, _ = x.shape
            for li, blk in enumerate(getattr(model, "blocks", [])):
                h = blk.ln1(x)              # pre-attn normalized hidden

                attn = blk.attn
                # Project to Q/K/V exactly like the module (pre-RoPE for simplicity)
                q = attn.wq(h).view(B, T, attn.n_head,   attn.d_head).transpose(1, 2)      # (B,H,T,D)
                k = attn.wk(h).view(B, T, attn.n_kv_head, attn.d_head).transpose(1, 2)     # (B,Hk,T,D)
                v = attn.wv(h).view(B, T, attn.n_kv_head, attn.d_head).transpose(1, 2)     # (B,Hk,T,D)

                # Take a tiny slice to keep logs light
                q1 = q[:1, :1].contiguous().view(-1).float().cpu()
                k1 = k[:1, :1].contiguous().view(-1).float().cpu()
                v1 = v[:1, :1].contiguous().view(-1).float().cpu()

                # Drop non-finite (defensive)
                q1 = q1[torch.isfinite(q1)]
                k1 = k1[torch.isfinite(k1)]
                v1 = v1[torch.isfinite(v1)]

                if q1.numel() > 0: logger.hist(f"qkv/block{li}/q_hist", q1, step)
                if k1.numel() > 0: logger.hist(f"qkv/block{li}/k_hist", k1, step)
                if v1.numel() > 0: logger.hist(f"qkv/block{li}/v_hist", v1, step)

                # Optional small scalars (norms) that show up on Time Series
                # (root-mean-square of the values = typical magnitude)
                if q1.numel(): logger.log(step=step, **{f"qkv/block{li}/q_l2_mean": float(q1.square().mean().sqrt())})
                if k1.numel(): logger.log(step=step, **{f"qkv/block{li}/k_l2_mean": float(k1.square().mean().sqrt())})
                if v1.numel(): logger.log(step=step, **{f"qkv/block{li}/v_l2_mean": float(v1.square().mean().sqrt())})

                # Advance x to next block with a CHEAP approximation to avoid doubling full compute:
                # use the model's own FFN path only; skip re-running attention (we're only logging pre-attn stats).
                x = x + blk.ffn(blk.ln2(x))

    except Exception as e:
        print(f"[qkv] logging failed: {e}")


def _log_runtime(logger, step: int, it_t0: float, xb, device):
    # Throughput (tokens/sec), step time and GPU memory: the main performance metrics for training.
    try:
        dt = time.time() - it_t0
        toks = int(xb.numel())   # tokens in this micro-batch = B * T
        toks_per_s = toks / max(dt, 1e-6)
        mem = torch.cuda.memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0
        logger.log(step=step, **{
            "sys/throughput_tokens_per_s": toks_per_s,
            "sys/step_time_s": dt,
            "sys/gpu_mem_alloc_mb": mem
        })
    except Exception:
        pass

def _log_samples_tb(logger, model, tok, xb, device, step: int, max_new_tokens: int = 64):
    # Generate a continuation of the first training sequence and log it as text — lets you
    # literally read how the model's output improves over training.
    if not _is_tb(logger): return
    if tok is None: return
    try:
        model.eval()
        with torch.no_grad():
            out = model.generate(xb[:1].to(device), max_new_tokens=max_new_tokens, temperature=1.0, top_k=50)
        model.train()
        text = tok.decode(out[0].tolist())
        logger.text("samples/generation", text, step)
    except Exception:
        pass
# ---------------------------------------------------------------------- #

def _extract_config_from_model(model) -> dict:
    """
    Best-effort extraction of GPTModern-like config including GQA fields.
    """
    # Reverse-engineer the constructor arguments from a live model, so a checkpoint is
    # self-describing and the exact architecture can be rebuilt at load time.
    cfg = {}
    try:
        tok_emb = getattr(model, "tok_emb", None)
        blocks = getattr(model, "blocks", None)
        if tok_emb is None or not blocks:
            return cfg

        try:
            from swiglu import SwiGLU  # optional
        except Exception:
            class SwiGLU: pass

        cfg["vocab_size"] = int(tok_emb.num_embeddings)   # rows of the embedding table = vocab size
        cfg["block_size"]  = int(getattr(model, "block_size", 0) or 0)
        cfg["n_layer"]     = int(len(blocks))

        first_blk = blocks[0]
        attn = getattr(first_blk, "attn", None)
        if attn is None:
            return cfg

        # Heads & dims
        cfg["n_head"]   = int(getattr(attn, "n_head"))
        d_head          = int(getattr(attn, "d_head"))
        cfg["n_embd"]   = int(cfg["n_head"] * d_head)
        cfg["n_kv_head"]= int(getattr(attn, "n_kv_head", cfg["n_head"]))  # default to MHA

        # Dropout (if present)
        drop = getattr(attn, "dropout", None)
        cfg["dropout"] = float(getattr(drop, "p", 0.0)) if drop is not None else 0.0

        # Norm/FFN style
        # GPTModern uses Identity as the final norm when RMSNorm is on, so that reveals the flag.
        cfg["use_rmsnorm"] = isinstance(getattr(model, "ln_f", None), nn.Identity)
        cfg["use_swiglu"]  = isinstance(getattr(first_blk, "ffn", None), SwiGLU)

        # Positional / attention tricks
        # (the attention module stores RoPE as `use_rope`, so "rope" is not actually found here
        #  and GPTModern's default rope=True is used on rebuild)
        for k in ("rope", "max_pos", "sliding_window", "attention_sink"):
            if hasattr(attn, k):
                val = getattr(attn, k)
                cfg[k] = int(val) if isinstance(val, bool) else val
    except Exception:
        return {}
    return cfg

def _verify_model_matches(model, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, message)."""
    # Compare the checkpoint's recorded architecture to the model we're loading into, and fail
    # with a readable message instead of a cryptic tensor-shape error.
    expected = {
        "block_size": cfg.get("block_size"),
        "n_layer":    cfg.get("n_layer"),
        "n_head":     cfg.get("n_head"),
        "n_embd":     cfg.get("n_embd"),
        "vocab_size": cfg.get("vocab_size"),
        "n_kv_head":  cfg.get("n_kv_head", cfg.get("n_head")),
    }
    got = {
        "block_size": int(getattr(model, "block_size", -1)),
        "n_layer":    int(len(model.blocks)),
        "vocab_size": int(model.tok_emb.num_embeddings),
    }
    first_blk = model.blocks[0]
    got.update({
        "n_head":     int(first_blk.attn.n_head),
        "n_embd":     int(first_blk.attn.n_head * first_blk.attn.d_head),
        "n_kv_head":  int(getattr(first_blk.attn, "n_kv_head", first_blk.attn.n_head)),
    })
    diffs = [f"{k}: ckpt={expected[k]} vs model={got[k]}" for k in expected if expected[k] != got[k]]
    if diffs:
        return False, "Architecture mismatch:\n  " + "\n  ".join(diffs)
    return True, "ok"


def save_checkpoint(model, optimizer, scheduler, amp, step: int, out_dir: str,
                    tokenizer_dir: str | None = None, config: dict | None = None):
    # A resumable checkpoint needs MORE than the weights:
    #  - optimizer state: AdamW keeps running averages of gradients (momentum) per parameter;
    #    losing them makes training jump when resumed
    #  - scheduler state: where we are on the LR curve
    #  - AMP scaler state: the current loss-scale factor
    #  - step counter, model config, and which tokenizer was used (ids are meaningless without it)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Prefer the model’s own config if available (e.g., a dict or dataclass with __dict__/asdict)
    if hasattr(model, "config"):
        cfg_obj = model.config
        cfg = dict(cfg_obj) if isinstance(cfg_obj, dict) else getattr(cfg_obj, "__dict__", None) or _extract_config_from_model(model)
    else:
        cfg = config if config is not None else _extract_config_from_model(model)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        # WarmupCosineLR has no state_dict(), so this is None in this repo and the LR schedule
        # restarts from step 0 (incl. warmup) on resume.
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "amp_scaler": amp.scaler.state_dict() if amp and getattr(amp, "scaler", None) else None,
        "step": int(step),
        "config": cfg,   # ← always write config
        "version": "part4-v2",
    }, out / DEF_NAME)

    # Store the tokenizer's location next to the checkpoint so sample.py / resume can find it.
    if tokenizer_dir is not None:
        (out / "tokenizer_dir.txt").write_text(tokenizer_dir)



def load_checkpoint(model, path: str, optimizer=None, scheduler=None, amp=None, strict: bool = True):
    # Load on CPU first (works on any machine); load_state_dict copies into the model's own device.
    ckpt = torch.load(path, map_location="cpu")

    cfg = ckpt.get("config")
    if cfg:
        ok, msg = _verify_model_matches(model, cfg)
        if not ok:
            raise RuntimeError(msg + "\nRebuild the model with this config, or load with strict=False.")
    else:
        # Legacy checkpoint without config: strongly encourage a rebuild step elsewhere
        print("[compat] Warning: checkpoint has no config; cannot verify architecture.")

    # strict=True: every parameter name in the checkpoint must match the model exactly.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch:\n  missing: {missing}\n  unexpected: {unexpected}")

    # Restore training state so the run continues as if it never stopped.
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if amp is not None and ckpt.get("amp_scaler") is not None and getattr(amp, "scaler", None):
        amp.scaler.load_state_dict(ckpt["amp_scaler"])

    return ckpt.get("step", 0)


# ----------------------------- checkpoint/save utils ----------------------------- #
def checkpoint_paths(out_dir: Path, step: int):
    # e.g. model_step0000300.pt (zero-padded so they sort correctly) and model_last.pt
    return out_dir / f"model_step{step:07d}.pt", out_dir / "model_last.pt"

def atomic_save_all(model, optim, sched, amp, step: int, out_dir: Path,
                    tok_dir: str | None, keep_last_k: int, config: dict):
    """Write model_last.pt (with config) + a rolling per-step copy."""
    save_checkpoint(model, optim, sched, amp, step, str(out_dir), tok_dir, config=config)  # writes model_last.pt
    per_step, last = checkpoint_paths(out_dir, step)
    try:
        shutil.copy2(last, per_step)
    except Exception:
        pass
    # GC old per-step checkpoints
    # (checkpoints can be huge — keep only the newest keep_last_k step files)
    try:
        ckpts = sorted(out_dir.glob("model_step*.pt"))
        for old in ckpts[:-keep_last_k]:
            old.unlink(missing_ok=True)
    except Exception:
        pass