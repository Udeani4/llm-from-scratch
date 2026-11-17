# LLM from Scratch — Project Description

## 1. What this project is

This repository is a hands-on curriculum that builds a small large language model (LLM) in **PyTorch**, from the first attention equation up to reinforcement-learning alignment. It does not use the Hugging Face `transformers` library, the `Trainer` API, or any pre-built model. Every component is written by hand: attention, positional encodings, the training loop, sampling, KV caching, mixture-of-experts, instruction fine-tuning, reward modelling, PPO and GRPO.

The project is split into nine numbered folders (`part_1` … `part_9`). Each folder covers one stage of the modern LLM pipeline. Together they reproduce, at toy scale, the recipe behind models like ChatGPT, Llama and DeepSeek:

```
                 ┌──────────────────────── ARCHITECTURE ────────────────────────┐
                 │ Part 1: Transformer basics   Part 3: Modern (Llama-style)    │
                 │ Part 5: Mixture-of-Experts (optional add-on)                 │
                 └──────────────────────────────────────────────────────────────┘
                                             │
 PRETRAINING           FINE-TUNING                     ALIGNMENT (RLHF)
 ───────────           ───────────                     ────────────────
 Part 2: tiny GPT  →   Part 6: SFT on instructions  →  Part 7: reward model
 Part 4: scaled-up     (prompt → response)              Part 8: PPO   ─┐
 training (BPE, AMP,                                    Part 9: GRPO  ─┴→ aligned policy
 LR schedule, ckpt)
```

The models are deliberately tiny (2–6 layers, 128–512 hidden units, 256-token context), so everything runs on a laptop CPU or a single consumer GPU in minutes. The point is to understand the **mechanics**, not to get a useful chatbot. The outputs of the later parts will be mostly nonsense, and that is expected at this scale.

### 1.1 Technology

| Item | Detail |
|---|---|
| Language | Python 3.11 (per README: `conda create -n llm_from_scratch python=3.11`) |
| Core library | `torch==2.8.0` |
| Tokenizer library | `tokenizers==0.21.4` (Hugging Face, Rust-backed BPE; used from Part 4 on) |
| Datasets | `datasets==4.0.0`, which downloads `tatsu-lab/alpaca` (Part 6, 8, 9) and `Anthropic/hh-rlhf` (Part 7) |
| Logging | `tensorboard==2.20.0` (optional `wandb`) |
| Testing | `pytest==8.4.1` |
| Plotting | `matplotlib` (Part 1 attention heatmaps) |

Install with `pip install -r requirements.txt`.

### 1.2 Repository conventions

These conventions apply to every part. Knowing them makes the code much easier to follow.

1. **Each part is self-contained and run from inside its own folder.** Imports are local (`from multi_head import ...`), so you must `cd part_N` first. Tests are also run from there (`python -m pytest -q tests/...`); pytest's rootdir handling puts the part folder on the import path.
2. **Each part has an `orchestrator.py`.** It is a small driver that runs the unit tests and, with `--demo`, a short end-to-end demo. It uses `subprocess.run(..., cwd=ROOT)` so every command executes inside the part's folder and aborts on the first failure.
3. **Later parts reuse earlier parts through `sys.path` manipulation.** Code from earlier folders is not copied. Instead the later files do things like:
   ```python
   sys.path.append(str(Path(__file__).resolve().parents[1] / 'part_3'))
   from model_modern import GPTModern
   ```
   `part_3/model_modern.py` also inserts the **repository root** into `sys.path` when it is imported. That side effect is what lets Parts 8 and 9 write `from part_6.formatters import ...`.
4. **Checkpoints are dictionaries** of the form `{'model': state_dict, 'config': {...}, ...}`. The `config` holds the constructor arguments, so a loader can rebuild the exact architecture before loading the weights.
5. **Pipeline artifacts live in `runs/` folders** inside each part, for example `part_4/runs/part4-demo/model_last.pt` and `part_4/runs/part4-demo/tokenizer/`. Later demos refer to these paths directly, so **the demos must be run in order** (4 → 6 → 7 → 8/9).

### 1.3 Tensor shape notation used throughout

| Symbol | Meaning |
|---|---|
| `B` | batch size (number of sequences processed together) |
| `T` | sequence length (number of tokens) |
| `C` / `d_model` / `n_embd` | hidden size: the length of each token's vector |
| `H` / `n_head` | number of attention heads |
| `D` / `d_head` | per-head dimension = `C / H` |
| `Hk` / `n_kv_head` | number of key/value heads (grouped-query attention) |
| `V` / `vocab_size` | number of distinct tokens |
| `E` | number of experts (MoE) |

---

## 2. Part 1: Core Transformer Architecture

**Goal:** build every piece of a transformer block by hand and check the shapes and math.

### 2.1 Files

| File | Role |
|---|---|
| `pos_encoding.py` | Learned and sinusoidal positional encodings |
| `attn_numpy_demo.py` | Self-attention computed in plain NumPy on a 3-token example, printing every intermediate |
| `attn_mask.py` | `causal_mask(T)`: the "no looking at the future" mask |
| `single_head.py` | One attention head in PyTorch |
| `multi_head.py` | Multi-head self-attention with optional shape printing |
| `ffn.py` | Feed-forward network (Linear → GELU → Linear) |
| `block.py` | A full pre-LayerNorm transformer block |
| `demo_mha_shapes.py` | Re-runs multi-head attention by hand and logs every shape to `out/mha_shapes.txt` |
| `demo_visualize_multi_head.py` + `vis_utils.py` | Saves a grid of per-head attention heatmaps to `out/` |
| `tests/` | Checks that the PyTorch head matches the NumPy example and that the mask is upper-triangular |

### 2.2 How it works

**Positional encodings.** Attention is order-blind: shuffling the input tokens just shuffles the outputs. To give the model word-order information, a position-dependent vector is **added** to each token embedding.
- *Learned* (`LearnedPositionalEncoding`): an `nn.Embedding(max_len, d_model)` table, one trainable vector per position (GPT-2 style). It cannot extrapolate past `max_len`.
- *Sinusoidal* (`SinusoidalPositionalEncoding`): fixed sine/cosine waves at geometrically spaced frequencies, `1 / 10000^(2i/d_model)`. These are stored with `register_buffer`, so they move with the model but are not trained.

**Scaled dot-product attention** (the core equation, shown step by step in `attn_numpy_demo.py`):

```
Q = X·Wq   K = X·Wk   V = X·Wv              # (B,T,d_k) each
scores  = Q·Kᵀ / √d_k                        # (B,T,T)   how well query i matches key j
scores  = masked_fill(future positions, -∞)  # causal mask
weights = softmax(scores, dim=-1)            # each row sums to 1
output  = weights · V                        # (B,T,d_k) weighted mix of values
```

- *Query/Key/Value intuition:* a query is "what this token is looking for", a key is "what this token offers to be found by", and a value is "the information handed over if attended to".
- *Why √d_k:* dot products grow with dimension. Without the scaling, softmax saturates into a near one-hot distribution and gradients vanish.
- *Why the causal mask:* the model is trained to predict the next token. If position *i* could see position *i+1*, it would simply copy the answer. `torch.triu(ones, diagonal=1)` marks exactly the (query, future-key) pairs, and `-inf` becomes probability 0 after softmax.
- *Numerically stable softmax:* the NumPy demo subtracts the row max before `exp`, which is the standard overflow guard.

**Multi-head attention** (`multi_head.py`). A single fused `nn.Linear(d_model, 3*d_model)` produces Q, K and V for all heads at once. The result is then reshaped:

```
(B,T,3C) → view (B,T,3,H,D) → unbind → q,k,v (B,T,H,D) → transpose → (B,H,T,D)
```

Putting `H` before `T` means `matmul` treats `(B, H)` as batch dimensions, so every head runs independently in parallel. Afterwards the heads are merged with `transpose(1,2).contiguous().view(B,T,C)`. The `.contiguous()` call is required because `view` needs contiguous memory after a transpose. A final `proj` linear layer mixes information across heads. Multiple heads let the model track several kinds of relationship at once, for example one head following the previous token and another following the sentence subject.

**Feed-forward network** (`ffn.py`). This is a per-token MLP: expand to `4×d_model`, apply GELU, project back. Attention moves information *between* tokens, while the FFN processes each token *individually*. Most of a transformer's parameters live here.

**Transformer block** (`block.py`) uses the **pre-LN** layout:
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```
The residual additions (`x + ...`) give gradients a direct path through the network, which is what makes deep stacks trainable. LayerNorm keeps each token vector at zero mean and unit variance. Pre-LN (normalise before the sub-layer) trains more stably than the original post-LN layout.

### 2.3 Analysis

- The NumPy demo and the PyTorch head use the same fixed weights, and `tests/test_attn_math.py` loads them into `nn.Linear`. That test teaches a subtle point: `nn.Linear` stores weights as `(out, in)` and computes `x @ W.T`, so the NumPy matrices must be transposed.
- The test only checks shape and finiteness. It does not compare numbers against the NumPy output, so it is a smoke test rather than an exact correctness test.
- `demo_mha_shapes.py` re-implements attention by hand without the causal mask, because it is purely about shapes.
- `vis_utils.py` heatmaps of a random-weight model will show lower-triangular patterns (from the mask) but no meaningful structure.

---

## 3. Part 2: Training a Tiny LLM

**Goal:** turn the block from Part 1 into a complete GPT and train it from scratch on a text file.

### 3.1 Files

| File | Role |
|---|---|
| `tokenizer.py` | `ByteTokenizer`: UTF-8 bytes as tokens (vocabulary of 256) |
| `dataset.py` | `ByteDataset`: random `(x, y)` windows with the next-token shift |
| `model_gpt.py` | `GPT`: embeddings + N blocks + LM head, plus `generate()` |
| `utils.py` | `top_k_top_p_filtering` for sampling |
| `train.py` | Training loop with eval, checkpointing, sampling, optional AMP and `torch.compile` |
| `sample.py` | Load a checkpoint and generate text |
| `eval_loss.py` | Average validation loss of a checkpoint |
| `tiny.txt`, `tiny_hi.txt` | Small training corpora |

### 3.2 How it works

**Tokenization.** A model only sees integers, so text must be converted into ids. The byte tokenizer is the simplest possible scheme: `"hi"` → `b"hi"` → `[104, 105]`. It can encode any text and has a tiny vocabulary, but sequences are long (one token per byte, two or more for non-ASCII characters). Decoding uses `errors='ignore'`, because a model can generate byte sequences that are not valid UTF-8.

**Dataset and the next-token objective.** The whole file is read as one long tensor of byte ids and split 90/10 into train and validation. `get_batch` picks random offsets and returns:
```
x = data[i     : i+block_size]
y = data[i+1   : i+1+block_size]      # x shifted left by one
```
`y[t]` is the token that follows `x[t]`. One window therefore yields `block_size` training examples at once: predict token 1 from token 0, token 2 from tokens 0–1, and so on. Thanks to the causal mask, every position learns simultaneously. This shift is the entire pretraining objective of a GPT.

**Model** (`model_gpt.py`):
```
idx (B,T) ─► tok_emb + pos_emb ─► dropout ─► Block × n_layer ─► LayerNorm ─► head ─► logits (B,T,V)
```
- Attention is delegated to `F.scaled_dot_product_attention(..., is_causal=True)`. This fused PyTorch kernel (it uses FlashAttention when available) computes the same thing as Part 1, faster.
- Weights use GPT-2 initialisation (normal with std 0.02), which keeps the initial logits small so training starts stably.
- **Loss** is cross-entropy over the flattened `(B·T, V)` logits against the `(B·T)` targets. It equals `-log P(correct next token)`, averaged. A useful reference point: a model that guesses uniformly over 256 bytes has loss `ln 256 ≈ 5.55`.

**Training loop** (`train.py`). This is the standard loop, written by hand:
1. Sample a batch.
2. Run the forward pass under `autocast` (fp16 when `--amp` is set).
3. Call `zero_grad`.
4. Backward through `GradScaler` (which prevents fp16 gradient underflow).
5. Unscale, then **clip gradients** to norm 1.0 (protects against spikes).
6. Take the optimizer step.

The optimizer is **AdamW** with `betas=(0.9, 0.95)` and weight decay 0.1, the usual LLM settings. Every `eval_interval` steps, `estimate_loss` averages the loss over many batches for both splits in `eval()` mode. The checkpoint with the **best validation loss** is saved, which is a simple form of early stopping. Every `sample_every` steps the model continues a random training snippet so you can watch the text improve.

**Sampling** (`generate` + `utils.py`). Generation is autoregressive: run the model, take the logits at the **last** position, apply temperature, filter, convert to probabilities, sample one token, append it, and repeat.
- *Temperature:* dividing logits by T < 1 sharpens the distribution (more deterministic); T > 1 flattens it (more random).
- *Top-k:* keep only the k highest-scoring tokens.
- *Top-p (nucleus):* sort the tokens and keep the smallest set whose probabilities add up to at least p. This adapts to the model's confidence: few candidates when it is sure, many when it is not.
- Both filters set excluded logits to `-inf`. Without filtering, the long tail of individually unlikely tokens adds up to a real chance of sampling garbage.
- The context is cropped to the last `block_size` tokens, because the learned position table has only `block_size` entries.

### 3.3 Analysis

- `generate()` re-runs the **whole** context for every new token, which is O(T) work per token. Part 3's KV cache fixes this.
- `generate()` calls `self.eval()` during training-time sampling. The next `estimate_loss()` switches back to train mode, so this is harmless but easy to miss.
- `model_final.pt` is saved **without** a config. `eval_loss.py` and `sample.py` fall back to default hyper-parameters in that case, which only work if the defaults match the trained model.
- The data split is by position (the last 10% of the file), so validation text is unseen but stylistically similar.

---

## 4. Part 3: Modernizing the Architecture

**Goal:** upgrade the GPT-2-style model to the design used by Llama, Mistral and Qwen, and make inference fast.

### 4.1 Files

| File | Role |
|---|---|
| `rmsnorm.py` | RMSNorm (replaces LayerNorm) |
| `rope_custom.py` | Rotary Positional Embedding: cos/sin cache plus the rotation |
| `swiglu.py` | SwiGLU gated feed-forward |
| `kv_cache.py` | `KVCache` dataclass and `RollingKV` bounded buffer |
| `attn_modern.py` | Attention with RoPE, KV cache, sliding window, attention sink, and GQA |
| `block_modern.py` | Block with switchable components (RMSNorm/LN, SwiGLU/GELU) |
| `model_modern.py` | `GPTModern`, the **base model reused by Parts 4, 6, 8 and 9** |
| `demo_generate.py` | Times generation with vs. without the KV cache |
| `tokenizer.py`, `utils.py` | Copies of the Part 2 versions, for self-containment |

### 4.2 How it works

**RMSNorm.** `y = x / sqrt(mean(x²) + eps) * g`. Compared with LayerNorm it skips the mean subtraction and the bias. It is cheaper and works as well in practice. The gain `g` is initialised to ones.

**RoPE (Rotary Positional Embedding).** Instead of *adding* a position vector to the embeddings, RoPE *rotates* the query and key vectors:
- Each head vector is split into pairs of numbers, treated as 2-D points.
- Pair *i* at position *p* is rotated by the angle `p · θᵢ`, where `θᵢ = 1/10000^(2i/D)`.
- The rotation is the standard 2-D one: `x1' = x1·cos − x2·sin`, `x2' = x1·sin + x2·cos`. It is implemented with even/odd slicing (`x[..., ::2]`, `x[..., 1::2]`).
- Because rotating q by `angle(m)` and k by `angle(n)` makes `q·k` depend only on `m − n`, attention scores encode **relative** position. Values (V) are not rotated.

`RoPECache` precomputes the cos/sin tables and doubles them on demand if a position exceeds `max_pos`. `GPTModern` therefore has no learned positional table; the `pos_emb` line is commented out.

**SwiGLU.** `out = W3( W1(x) ⊙ SiLU(W2(x)) )`. One branch computes values and the other computes a smooth gate that controls how much of each value passes through. It gives better quality than GELU at similar compute. The repo keeps `inner = 4×dim`; Llama shrinks this to about ⅔ of that to keep the parameter count equal.

**KV cache.** In causal attention, the keys and values of past tokens never change. So during generation:
- **Prefill:** the first call processes the whole prompt and stores K and V for every layer.
- **Decode:** each later call feeds **only the newest token**, computes its q/k/v, appends k/v to the cache, and attends over the full cache.

This turns per-token cost from re-processing the whole context into processing one token. Two details make it correct:
- `start_pos` tells RoPE the absolute position of the new token, which equals the cache length.
- `is_causal = kv_cache is None`. A mask is needed on the prefill pass, but when a single new query attends to cached keys, everything is in the past and no mask is needed.

`GPTModern.forward` returns `(logits, loss, new_caches)`, one `KVCache` per layer. `generate()` threads these caches through the loop. `generate_nocache()` is the slow reference path and prints the top-10 probabilities at each step.

**Grouped-Query Attention (GQA).** Q has `n_head` heads, but K and V have only `n_kv_head` heads. Each KV head is shared by `group_size = n_head / n_kv_head` query heads, expanded with `repeat_interleave` just before attention. The cache stores the **compact** `Hk` heads, and that is the memory saving: KV-cache size is the main limit on long-context inference. `n_kv_head == n_head` is ordinary multi-head attention; `n_kv_head == 1` is multi-query attention.

**Sliding window + attention sink.** To bound attention cost on long sequences, only the first `attention_sink` tokens plus the most recent `sliding_window` tokens are attended to (as in Mistral and StreamingLLM). Sink tokens matter because models learn to "park" excess attention on the first tokens; dropping them makes generation quality collapse. `RollingKV` in `kv_cache.py` shows the same idea as a bounded-memory buffer, and its test checks that the length never exceeds `sink + window`.

### 4.3 Analysis

- **No final norm with RMSNorm:** with `use_rmsnorm=True`, `ln_f` is `nn.Identity()`, so no final normalisation is applied before the LM head. Real Llama models do apply one. It works at this scale but is a deviation.
- **Sliding window only limits compute, not memory:** in `attn_modern.py` the window crops what is *attended to*, but the full uncropped K/V are still stored in the returned cache. Memory still grows without bound. On a long prompt (prefill longer than the window) with `is_causal=True`, the query length and the cropped key length differ, and SDPA aligns the causal mask from the top-left, so the masking is not exactly right in that case. `RollingKV` is the bounded version but is not wired into the model.
- **Unused parameters:** the `sliding_window`/`attention_sink` arguments of `generate()` are ignored, because those settings are fixed on the attention layers when the model is built.
- **Default end-of-sequence id:** `generate()` has `eos_id=1`, added for Part 6, where BPE id 1 is `</s>`. With the byte tokenizer, byte 1 is a control character that is almost never produced, so it is harmless there.
- **Misleading flag name:** `demo_generate.py`'s `--group_size` is actually passed as `n_kv_head`.
- **Expected demo output:** the demo runs an **untrained** model, so its text is gibberish. The point is the timing, and that greedy decoding (`temperature=0`) gives the same output with and without the cache.

---

## 5. Part 4: Scaling Up

**Goal:** add the engineering that makes real pretraining practical: a better tokenizer, memory and speed tricks, a learning-rate schedule, resumable checkpoints, and logging. The model is `GPTModern` from Part 3.

### 5.1 Files

| File | Role |
|---|---|
| `tokenizer_bpe.py` | `BPETokenizer` wrapper around Hugging Face `ByteLevelBPETokenizer` |
| `dataset_bpe.py` | `TextBPEBuffer` dataset + `make_loader` (`DataLoader`) |
| `amp_accum.py` | `AmpGrad`: mixed precision + gradient accumulation |
| `lr_scheduler.py` | `WarmupCosineLR` |
| `checkpointing.py` | Save/load full training state, plus TensorBoard diagnostic helpers |
| `logger.py` | `TBLogger` / `WBLogger` / `NoopLogger` behind one `.log()` interface |
| `train.py` | The scaled-up training loop with automatic resume |
| `sample.py` | Generation from a Part 4 checkpoint |

### 5.2 How it works

**BPE tokenization.** Byte-Pair Encoding starts from the 256 bytes and repeatedly merges the most frequent adjacent pair into a new token (`t`+`h`→`th`, `th`+`e`→`the`, …) until `vocab_size` is reached. Common words become single tokens, while rare words split into pieces but stay encodable. Sequences come out roughly 3–4× shorter than with bytes, so far more text fits in the context window. Special tokens are reserved first, and their ids follow list order: `<s>`=0, `</s>`=1, `<pad>`=2, `<unk>`=3, `<mask>`=4. **These ids are relied on later:** `eos_id=1` stops generation, and id 2 is the padding id in Parts 6–9. The tokenizer is trained on the training data itself and saved (`vocab.json`, `merges.txt`, `tokenizer.json`, `bpe_meta.json`). `load()` uses `tokenizer.json`.

**Dataset.** The whole file is tokenized once into a 1-D id stream. Every start offset is one sample, using the same `x`/`y` shift as Part 2, and a `DataLoader` shuffles and batches them.

**Mixed precision (AMP).** Under `torch.cuda.amp.autocast`, matmuls run in fp16, which is about twice as fast and halves activation memory. Precision-sensitive operations (softmax, loss) stay in fp32. `GradScaler` multiplies the loss before backward so small fp16 gradients do not underflow, un-scales them before the step, and skips the step if it finds inf or NaN.

**Gradient accumulation.** `AmpGrad.backward` divides the loss by `accum` and calls backward. Gradients **add up** in `.grad` across calls, and the optimizer steps only every `accum` micro-batches. The effective batch becomes `batch_size × accum` without the memory cost. Dividing by `accum` makes the accumulated gradient equal to the gradient of the *mean* loss.

**Warmup + cosine LR.** The learning rate ramps linearly from about 0 to `base_lr` over the warmup steps, then follows half a cosine down to 0. Warmup avoids destabilising large updates while the weights are random and Adam's statistics are empty. Cosine decay lets the model settle at the end. Warmup is capped at 10% of total steps.

**Checkpointing and resume.** `save_checkpoint` stores the model weights, the **optimizer state** (AdamW's per-parameter momentum; losing it makes a resumed run jump), the scheduler, the AMP scaler, the step counter, the model config, and a pointer file `tokenizer_dir.txt`. `train.py` resumes automatically: if `out/model_last.pt` exists, it reloads the original tokenizer, refuses to continue if the vocabulary changed, verifies the architecture with `_verify_model_matches`, and restores all state. `atomic_save_all` also keeps a rolling set of `model_stepNNNNNNN.pt` copies (`keep_last_k`). A SIGINT/SIGTERM handler (Ctrl+C or a cluster pre-emption) sets a flag, and the loop saves cleanly before exiting.

**Logging and diagnostics** (TensorBoard: `tensorboard --logdir runs/part4-demo`):
- loss and LR every 50 optimizer steps
- throughput (tokens/s), step time, GPU memory
- global parameter and gradient L2 norms
- Q/K/V histograms and RMS per block, to spot attention logits blowing up (a common cause of loss spikes)
- generated text samples, hyper-parameters, and the model graph

All logging calls are wrapped in `try/except` so a logging failure never kills training.

### 5.3 Analysis

This is the most "production-flavoured" part. A few real gaps are worth knowing:
- **Scheduler not saved:** `WarmupCosineLR` has no `state_dict()`, so the scheduler is never saved. On resume, the LR schedule restarts from step 0, including warmup.
- **Gradient norm logs NaN:** `_log_model_stats` runs right after `zero_grad(set_to_none=True)`, so all gradients are `None`.
- **No gradient clipping** in this loop, unlike Part 2.
- **Step budget mixes units:** `total_steps = min(args.steps, epochs * len(train_loader))` counts *micro-batches* from the loader against *optimizer* steps. With `--steps` set, `args.steps` is normally the binding limit.
- **The byte fallback is broken:** `make_loader` needs a tokenizer, so without `--bpe` it fails. In practice `--bpe` is required.
- **Vocabulary size may be overstated:** `BPETokenizer.load()` restores the *requested* `vocab_size` from metadata. On a tiny corpus the real vocabulary can be smaller. This is harmless (the embedding table is just larger than needed), but `vocab_size` is the target, not the actual size.
- **Graph logging switches mode:** `_maybe_log_graph_tb` calls `model.eval()` on the real model through the wrapper. Training then runs in eval mode, which only matters if dropout > 0 (it defaults to 0).
- **RoPE flag not recovered:** `_extract_config_from_model` looks for an attribute `rope`, but the attention layer stores it as `use_rope`. In practice the explicit config from `train.py` is used instead.
- **Demo resumes if output exists:** the orchestrator demo will *resume* if `runs/part4-demo/model_last.pt` already exists. Delete the folder for a fresh run.

---

## 6. Part 5: Mixture-of-Experts (MoE)

**Goal:** implement the sparse layer used by Mixtral, DeepSeek-V3 and similar models. This part is **standalone**: it is not plugged into the GPT pipeline.

### 6.1 Files

| File | Role |
|---|---|
| `experts.py` | `ExpertMLP`: one FFN (SwiGLU or GELU) |
| `gating.py` | `TopKGate`: router plus Switch-Transformer load-balancing loss |
| `moe.py` | `MoE`: dispatch tokens to experts and combine the outputs |
| `block_hybrid.py` | `HybridFFN`: `α·Dense(x) + (1−α)·MoE(x)` |
| `demo_moe.py` | Forward pass plus a histogram of how many tokens each expert got |
| `README.md` | Concept notes |

### 6.2 How it works

**Why MoE:** it decouples parameter count from compute. With E experts and top-1 routing, the layer has about E times the parameters of one FFN (more capacity) but each token pays the compute of only one FFN.

**Routing** (`TopKGate`). A linear layer scores all E experts for each token. Softmax turns the scores into probabilities, and `topk` keeps the best k. The kept probabilities become the mixing weights. They are not renormalised, so with k>1 they sum to ≤ 1. Gradients reach the router through these weights.

**Load-balancing auxiliary loss.** Left alone, the router collapses onto one or two favourite experts ("expert collapse"). The Switch loss is:
```
importance_e = mean over tokens of P(expert e)       # soft, differentiable
load_e       = fraction of tokens whose top-1 is e    # hard counts
aux          = E · Σ_e importance_e · load_e
```
It equals 1.0 under perfectly uniform routing and grows as routing concentrates. Only `importance` carries gradients, nudging probability away from overloaded experts. The caller must add `coef × aux` (about 0.01) to the main loss.

**Dispatch and combine** (`MoE.forward`). Tokens are flattened to `(B·T, C)`. Then, for each expert and each of the k slots, a boolean mask selects the tokens routed there. The expert runs only on those tokens, and its output is added back in place, weighted by the router probability. The loop over experts is written for clarity; production systems use batched scatter/gather and expert parallelism across GPUs.

**Hybrid FFN.** Every token always goes through a dense FFN, and a routed MoE output is blended in with a fixed `α`. This resembles the "shared expert" idea in DeepSeek-MoE: a reliable dense path plus specialised capacity.

### 6.3 Analysis

- Expert **capacity limits** and token dropping, used in Switch and GShard, are not implemented. Any number of tokens can go to one expert.
- The aux loss uses only the top-1 assignment, even when k=2.
- `demo_moe.py` uses random inputs and an untrained router, so the histogram reflects random initial routing. Its value is that it shows *how* imbalance is measured.
- There is no MoE-based GPT model in the repo. The obvious extension is to replace `block_modern.py`'s `self.ffn` with `MoE` and add `aux` to the training loss.

---

## 7. Part 6: Supervised Fine-Tuning (SFT)

**Goal:** turn the pretrained "text continuer" from Part 4 into an **instruction follower** by training on (instruction, response) pairs.

### 7.1 Files

| File | Role |
|---|---|
| `formatters.py` | The chat template |
| `dataset_sft.py` | Loads Alpaca (`tatsu-lab/alpaca`), with a 3-example fallback |
| `collator_sft.py` | `SFTCollator`: tokenize, shift and **mask** the labels, pad |
| `curriculum.py` | `LengthCurriculum`: shortest prompts first |
| `evaluate.py` | Exact-match and token-F1 metrics |
| `train_sft.py` | SFT loop starting from the Part 4 checkpoint |
| `sample_sft.py` | Ask the fine-tuned model a question |

### 7.2 How it works

**The template:**
```
<s>
### Instruction:
{instruction}

### Response:
{response}</s>
```
During SFT the model learns that text after `### Response:` answers the instruction, and that `</s>` means "done". At inference, `sample_sft.py` feeds everything up to `### Response:\n`, so the most natural continuation is an answer, and `generate()` stops when the model emits `</s>` (id 1). Real chat models work the same way with their own special tokens.

**Loss masking.** This is the key idea of SFT. The loss is still next-token cross-entropy, but only on **response** tokens. The model should learn to answer, not to generate user prompts. In `collate`:
1. Tokenize the full text, and separately the prompt prefix, to find the prompt length `n_prompt`.
2. Build pre-shifted labels: `y[t] = ids[t+1]`, with `y[-1] = -100`. Pre-shifting is needed because `GPTModern.forward` does **not** shift internally.
3. Set `y[0 : n_prompt-1] = -100`. Position `n_prompt-1` (the last prompt token) is kept, because it predicts the first response token.
4. Pad inputs with id 2 (`<pad>`) and labels with `-100`.

`-100` is `F.cross_entropy`'s default `ignore_index`, so those positions contribute no loss or gradient. Padding sits *after* the real tokens, and with causal attention real tokens never attend to it.

**Curriculum.** Examples are sorted by prompt length, so the model sees easier (shorter) examples first, like a school syllabus.

**Training.** The Part 4 weights and the **same tokenizer** are loaded, because token ids must line up with the learned embeddings. Then plain AdamW steps are taken over batches drawn from the curriculum list, wrapping around at the end. The result, `runs/sft-demo/model_last.pt`, becomes the starting **policy** for RLHF.

**Evaluation metrics** (`evaluate.py`). These are SQuAD-style. Both texts are normalised (lowercase, punctuation stripped), then:
- *Exact match* is 1 if the strings are identical.
- *Token F1* is the harmonic mean of word-overlap precision and recall.

They are suited to short factual answers and say little about open-ended quality, which motivates the learned reward model in Part 7.

### 7.3 Analysis

- **Tiny dataset:** the demo trains on only 24 examples for 300 steps (100 batches of 8, about 100 epochs). The model will memorise them. Real SFT uses thousands to millions of examples for 1–3 epochs.
- **Tokenizer boundary assumption:** the masking assumes the prefix tokenizes identically on its own and inside the full text. That usually holds here because the prefix ends in a newline, but BPE merges can cross boundaries in general.
- **Config ignored:** `train_sft.py` reads the checkpoint's config but ignores it. The model is built from CLI flags, which must match Part 4's demo model (2 layers, 2 heads, 128-dim).
- **Unused module:** `evaluate.py` is not called by any script; it is a library for your own experiments.
- **Output includes the prompt:** `sample_sft.py` prints prompt + answer. Decoding `out_ids[orig_len:]` would print the answer only.
- **Weak masking test:** `test_masking.py` only checks that some `-100`s exist. It does not verify the exact boundary.

---

## 8. Part 7: Reward Modeling

**Goal:** train a model that reads a (prompt, response) and outputs a **single score**, how good a human would judge it. This is the learned stand-in for human raters that RLHF optimises against.

### 8.1 Files

| File | Role |
|---|---|
| `data_prefs.py` | Loads `Anthropic/hh-rlhf` (chosen/rejected pairs), with a toy fallback |
| `collator_rm.py` | `PairCollator`: tokenizes each pair into `pos` and `neg` tensors |
| `model_reward.py` | `RewardModel`: transformer encoder → mean-pool → scalar |
| `loss_reward.py` | Bradley–Terry and margin-ranking losses |
| `train_rm.py` | Training loop |
| `eval_rm.py` | Pairwise accuracy |

### 8.2 How it works

**Preference data.** Humans find it hard to give absolute scores but easy to compare: "answer A is better than B". Each example is therefore `(prompt, chosen, rejected)`. HH-RLHF stores full conversations in `chosen` and `rejected`, so the prompt field is left empty.

**Model.** Token + learned position embeddings feed PyTorch's built-in `nn.TransformerEncoder`. This is **bidirectional**: there is no causal mask, which is fine because the model never generates. Padding (id 2) is excluded from attention with `src_key_padding_mask`. The token vectors are **mean-pooled** over real tokens into one vector, and a `Linear(n_embd, 1)` produces the reward.

**Loss.** Only the score *difference* has meaning:
- **Bradley–Terry** (the default, used by InstructGPT and Llama-2): `P(chosen ≻ rejected) = σ(r_pos − r_neg)`, loss `= −log σ(r_pos − r_neg) = softplus(−(r_pos − r_neg))`. The softplus form is the numerically stable version. In effect this is logistic regression on the score gap.
- **Margin ranking:** `max(0, margin − (r_pos − r_neg))`. This hinge loss stops pushing once the chosen answer leads by the margin.

**Metric.** Pairwise accuracy is the fraction of pairs where `r_pos > r_neg`. 0.5 is chance. Real reward models on HH reach roughly 0.65–0.75, because human preferences are noisy.

### 8.3 Analysis

- **Separate small encoder:** production reward models are usually the SFT LLM itself, with its LM head replaced by a 1-output head and the score read from the **last token's** hidden state. This repo uses a separate small encoder trained from scratch with mean pooling. That is simpler, but it starts with no language knowledge and was not trained on the policy's own text distribution.
- **Truncation:** HH conversations are long and get truncated to 256 tokens, often cutting off the final assistant turn where the chosen and rejected answers actually differ.
- **Little data:** only 80 training pairs, so expect overfitting. The orchestrator evaluates on `train[:8]` (should be high) and `test[:8]` (the real generalisation check; expect near chance).
- **Wrong default split:** `eval_rm.py`'s default `--split val[:200]` does not exist in HH-RLHF, so it silently falls back to the 2 toy pairs.
- **Untrained tokenizer without `bpe_dir`:** if `bpe_dir` is `None` while `tokenizers` is installed, `PairCollator` holds an **untrained** BPE object and encoding fails. Always pass `--bpe_dir`.

---

## 9. Part 8: RLHF with PPO

**Goal:** improve the SFT policy by reinforcement learning against the reward model, using **Proximal Policy Optimization**, the algorithm behind InstructGPT and the original ChatGPT.

### 9.1 Files

| File | Role |
|---|---|
| `policy.py` | `PolicyWithValue`: `GPTModern` + value head |
| `rollout.py` | Tokenizer wrapper, log-prob helpers, KL estimate, prompt source |
| `ppo_loss.py` | Clipped policy loss, value loss, entropy, approximate KL |
| `train_ppo.py` | The RLHF loop |
| `eval_ppo.py` | Average reward-model score of the policy's answers |

### 9.2 The RL framing of a language model

| RL concept | LLM meaning |
|---|---|
| Policy π | The language model |
| State | Prompt + tokens generated so far |
| Action | Choosing the next token |
| Episode | Generating one full response |
| Reward | Reward-model score of the finished response, minus a KL penalty |
| Value V(s) | Predicted final reward from this state onward (the "critic") |
| Advantage | How much better an action turned out than V expected |

Three models are involved:
- the **policy** (trainable, initialised from SFT)
- the **reference** (a frozen copy of SFT)
- the **reward model** (frozen)

### 9.3 One PPO iteration (`train_ppo.py`)

1. **Rollout.** Take `batch_size` prompts, wrap them in the SFT template, and let the current policy generate responses (`temperature=0.2`, `top_k=3`, stopping at `</s>`).
2. **Score.** Re-format prompt + response in the template and run the reward model to get one scalar per response.
3. **Batch.** Right-pad the sequences with id 2. `mask` marks response tokens, and the reward sits on each sequence's **last token** (a sparse, end-of-episode reward).
4. **Log-probs.** `model_logprobs` computes `log π(x[t+1] | x[≤t])` for the policy and the reference. `act_mask = mask[:,1:]` keeps the positions whose *predicted* token is a response token. Everything is flattened to one entry per response token.
5. **KL-shaped reward.** `shaped_r = reward − kl_coef · (log π − log π_ref)` per token. The KL term penalises drifting from SFT, which is the main defence against **reward hacking** (degenerate text that fools the reward model).
6. **Advantage.** `adv = shaped_r − V`, normalised to zero mean and unit standard deviation.
7. **Update.** Recompute log-probs and values with gradients and minimise:
   ```
   ratio       = exp(new_logp − old_logp)
   policy_loss = −mean( min(ratio·adv, clip(ratio, 1−ε, 1+ε)·adv) )   # ε = 0.2
   value_loss  = MSE(V_new, returns)
   total       = policy_loss + 0.5·value_loss − ent_coef·entropy
   ```
   The clip is PPO's "trust region". Once a token's probability has moved more than about 20% in the advantage's direction, there is no further gain from pushing it, which keeps updates small and stable. Gradients are clipped to norm 1.0, and the LR is a small 1e-5.
8. **Diagnostics.** Two quantities are printed:
   - `KL_move`: how much this update changed the policy.
   - `KL_ref`: total drift from SFT. It should grow slowly; if it explodes, increase `kl_coef`.

**Evaluation** (`eval_ppo.py`). The script generates answers to 16 prompts and averages their reward-model scores. If RL worked, the average should be higher than for SFT. A higher score alone does not prove better answers, since the policy may be exploiting the reward model, so read the samples too.

### 9.4 Analysis

This is a faithful *skeleton* of RLHF-PPO with several large simplifications. Know them before drawing conclusions from the numbers:
- **No credit assignment over time.** `returns = shaped_r`, so each token's return is only its *own* reward. The reward-model score directly credits only the final token; earlier tokens see only their KL penalty. Proper PPO computes returns and advantages with **GAE**, summing discounted future rewards so every token shares credit. `--gamma` and `--lam` exist but are unused.
- **Single update epoch.** At update time `new_logp == old_logp`, so `ratio == 1` and the clip never activates. This is effectively vanilla policy gradient with a baseline. Real PPO runs several epochs and mini-batches per rollout, which is where clipping matters.
- **Toy value head.** It sits on the **logits** (`Linear(vocab, 1)`) instead of the final hidden state (`Linear(n_embd, 1)`). It is randomly initialised.
- **Rough entropy.** It is approximated as `−mean(log p(sampled token))` rather than computed from the full distribution. It is off by default (`ent_coef=0`).
- **No reward normalisation** across batches, even though the README mentions it.
- **Hardcoded path and unused output in eval:** the SFT reference path is fixed at `../part_6/runs/sft-demo/model_last.pt`. The reference model's responses are generated but neither scored nor printed. `--split` is unused.
- **Fragile command splitting:** `orchestrator.py` uses `cmd.split()` instead of `shlex.split`, so arguments cannot contain quoted spaces.

---

## 10. Part 9: RLHF with GRPO

**Goal:** the same alignment task with **Group Relative Policy Optimization** (from DeepSeekMath / DeepSeek-R1), which removes the value network entirely.

### 10.1 Files

| File | Role |
|---|---|
| `train_grpo.py` | GRPO loop |
| `grpo_loss.py` | `ppo_policy_only_losses`: clipped policy loss + KL term, no value loss |
| `policy.py`, `rollout.py`, `eval_ppo.py` | Identical copies of Part 8's |
| `tests/test_grpo_loss.py` | Loss sanity test |

### 10.2 How it works

The key insight: PPO needs a value head only to provide a **baseline** ("what reward did we expect?"). GRPO gets the baseline from **other samples of the same prompt** instead:

1. Take `P` prompts and generate **G completions for each** (`B = P × G` trajectories). Sampling uses `temperature=2` deliberately, so the G answers differ. If they were identical, every advantage would be zero and nothing would be learned.
2. Score each completion with the reward model.
3. **Group baseline.** For each prompt, compute the mean reward of its G completions. Each completion's advantage is `reward − group_mean`, meaning "was this answer better or worse than my other attempts at the same question?"
4. **Broadcast.** That single trajectory-level advantage is assigned to every token of the completion (`traj_id_for_token` maps flattened tokens back to their trajectory). The advantages are then normalised across the batch.
5. **Loss.** This is PPO's clipped policy objective with **no value loss**, plus an explicit KL term **in the loss** rather than in the reward:
   ```
   total = policy_loss + kl_coef · mean(new_logp − ref_logp)
   ```
   The KL is computed *with gradients* from the new log-probs, so minimising it pulls the policy back toward SFT.

### 10.3 PPO vs GRPO

| | PPO (Part 8) | GRPO (Part 9) |
|---|---|---|
| Baseline | Learned value head | Mean reward of G samples per prompt |
| Extra network | Yes (critic) | No |
| Samples per prompt | 1 | G (4 in the demo) |
| Credit unit | Per token (in principle, via GAE) | Whole trajectory, broadcast to its tokens |
| KL handling | Subtracted from the reward | Added to the loss |
| Main cost | Training/fitting the critic | G× more generation |

GRPO's appeal is simplicity and robustness: there is no critic to get wrong. Its main cost is generating several completions per prompt.

### 10.4 Analysis

- **Batch-wide normalisation:** the GRPO paper normalises by each group's standard deviation, `(r − mean_g)/std_g`. Here the advantages are mean-centred per group and then standardised **batch-wide**. That is close in spirit, but groups with tiny spread are not amplified.
- **Clipping inactive:** as in Part 8 there is one update per batch, so `ratio == 1` and the clip is inactive.
- **Unused variables:** `kl_tok` (computed per token before the update), `counts`, `offset` and `last_idx_list` are only for logging or bookkeeping. The comment mentions "KL shaping at trajectory level", but KL is actually applied only in the loss.
- **High temperature:** `temperature=2` with `top_k=3` gives diversity but noisy text. Reasoning-model setups typically use around 0.7–1.0 with many more samples.
- **Leftover value head:** the policy still carries an (unused, untrained) value head, because `PolicyWithValue` is reused.
- **Stale header:** the orchestrator's layout header was copied from Part 8 and mentions files and folders that do not exist in Part 9.

---

## 11. How the parts connect: end-to-end run order

Artifacts flow from one part to the next through fixed relative paths:

```
part_4/runs/part4-demo/tokenizer/      ← BPE tokenizer (used by Parts 6, 7, 8, 9)
part_4/runs/part4-demo/model_last.pt   ← pretrained base model
          │
          ▼
part_6/runs/sft-demo/model_last.pt     ← SFT policy (start point + reference for RL)
part_7/runs/rm-demo/model_last.pt      ← reward model
          │
          ▼
part_8/runs/ppo-demo/model_last.pt     ← PPO-aligned policy
part_9/runs/grpo-demo/model_last.pt    ← GRPO-aligned policy
```

Run the parts in this order. On Windows PowerShell, use `;` between commands:

```powershell
cd part_1; python orchestrator.py --visualize; cd ..   # architecture demos and tests
cd part_2; python orchestrator.py; cd ..               # tiny byte-level GPT
cd part_3; python orchestrator.py --demo; cd ..        # modern architecture, KV cache timing
cd part_4; python orchestrator.py --demo; cd ..        # BPE + pretraining  (required by 6–9)
cd part_5; python orchestrator.py --demo; cd ..        # MoE (independent)
cd part_6; python orchestrator.py --demo; cd ..        # SFT                (needs part 4)
cd part_7; python orchestrator.py --demo; cd ..        # reward model       (needs part 4 tokenizer)
cd part_8; python orchestrator.py --demo; cd ..        # PPO                (needs 4, 6, 7)
cd part_9; python orchestrator.py --demo; cd ..        # GRPO               (needs 4, 6, 7)
```

Parts 6–9 need internet access the first time, to download Alpaca and HH-RLHF. Without it they fall back to small built-in examples.

**Architecture compatibility.** The Part 4 demo trains a 2-layer, 2-head, 128-dim, 128-token-context model. Part 6 loads it with `--block_size 256` and the same layer/head/width settings. This works because `GPTModern` has no position table whose shape depends on `block_size`; RoPE makes the context length a runtime limit rather than a weight shape.

---

## 12. Overall assessment

### Strengths
- **Complete pipeline at readable scale.** Few educational repos go all the way from attention math to PPO and GRPO with one shared model class.
- **Modern components.** RoPE, RMSNorm, SwiGLU, GQA, KV cache, sliding window and attention sink, so the architecture matches 2023–2025 open models rather than GPT-2.
- **Real engineering concerns.** Mixed precision, gradient accumulation, LR schedules, strict resumable checkpoints, signal-safe saving and training-health diagnostics are all covered.
- **Every part is testable in isolation**, with a `tests/` folder and an orchestrator.

### Limitations, by design
- The models and data are far too small to produce coherent text. Losses and rewards show the *mechanics*, not quality.
- The RL parts are simplified: no GAE, a single update epoch, and a value head on logits. They demonstrate the structure of PPO and GRPO, not their full behaviour.
- The reward model is a separate small encoder rather than an LLM-based model, and it is trained on truncated conversations.

### Known issues found in the code

| Part | Issue | Effect |
|---|---|---|
| 3 | `ln_f` is `Identity` when RMSNorm is on | No final norm before the LM head |
| 3 | Sliding window crops attention but the full cache is still stored | Memory is not bounded; causal-mask alignment is wrong on long prefills |
| 4 | Scheduler state is never saved | LR warmup restarts on resume |
| 4 | Gradient norm is logged after `zero_grad` | `train/grad_global_l2` is always NaN |
| 4 | No gradient clipping in the loop | Less protection against loss spikes than Part 2 |
| 4 | Byte-level fallback cannot build a loader | `--bpe` is effectively required |
| 6 | 24 examples for ~100 epochs | The SFT model memorises the data |
| 7 | Default eval split `val` does not exist | Silently evaluates on 2 toy pairs |
| 7 | Long HH conversations are truncated to 256 tokens | The differing final turn is often cut off |
| 8 | `returns = shaped_r`, no GAE | Reward credits only the last token |
| 8, 9 | One update per rollout | PPO clipping never activates |
| 8, 9 | Hardcoded SFT path in `eval_ppo.py`; reference responses are unused | Eval only works with the default folder layout |
| 9 | Orchestrator header copied from Part 8 | Misleading file list |

### Natural next steps
1. Add GAE and multiple PPO epochs/mini-batches in Part 8, so clipping and per-token credit actually work.
2. Initialise the reward model from the SFT model and score from the last token.
3. Plug `MoE` into `TransformerBlockModern` and add the aux loss to Part 4's training loop.
4. Wire `RollingKV` into `attn_modern.py` so the sliding window bounds memory.
5. Save and restore the LR scheduler state, and move gradient logging before `zero_grad`.
6. Use `part_6/evaluate.py` to compare SFT, PPO and GRPO outputs against Alpaca reference answers.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Token** | Unit of text the model reads: a byte (Parts 1–3) or a BPE piece (Part 4+) |
| **Logits** | Raw, unnormalised scores over the vocabulary; softmax turns them into probabilities |
| **Context window / block_size** | Maximum number of tokens the model sees at once |
| **Autoregressive** | Generates one token at a time, each conditioned on all previous ones |
| **Causal mask** | Prevents attention to future positions |
| **Residual connection** | `x + f(x)`: adds a sub-layer's output back to its input |
| **Pretraining** | Next-token prediction on raw text |
| **SFT** | Supervised fine-tuning on instruction→response pairs, with the prompt masked out of the loss |
| **RLHF** | Reinforcement Learning from Human Feedback: optimise a policy against a reward model trained on human preferences |
| **Reward hacking** | The policy finds outputs that score highly with the reward model without actually being good |
| **KL divergence (to reference)** | How far the policy's token distribution has drifted from the SFT model; used as a penalty |
| **Advantage** | Reward relative to a baseline (a value head in PPO, the group mean in GRPO) |
| **KV cache** | Stored keys/values of past tokens, so generation does not recompute them |
| **GQA** | Grouped-query attention: fewer K/V heads than Q heads, for a smaller KV cache |
| **MoE** | Mixture-of-Experts: a router sends each token to a few of many expert FFNs |
| **AMP** | Automatic mixed precision: fp16 compute with fp32 safeguards |
