import torch, tempfile, os
import torch.nn as nn
from checkpointing import save_checkpoint, load_checkpoint

class Dummy(nn.Module):
    # Minimal model (one Linear) — this test is about the save/load plumbing, not the architecture.
    # It has no tok_emb/blocks, so the saved config is empty and the architecture check is skipped.
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(8,8)


def test_save_and_load(tmp_path):
    m = Dummy(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # Bare stand-ins for the scheduler and AmpGrad objects (only the attributes checkpointing touches).
    class S: pass
    sch = S(); sch.__dict__ = {'warmup_steps': 10, 'total_steps': 100, 'base_lr': 1e-3, 'step_num': 5}
    class A: pass
    amp = A(); amp.scaler = torch.cuda.amp.GradScaler(enabled=False)

    out = tmp_path/"chk"
    save_checkpoint(m, opt, sch, amp, step=123, out_dir=str(out), tokenizer_dir=None)

    # Fresh objects, as if restarting the training process.
    m2 = Dummy(); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sch2 = S(); sch2.__dict__ = {'warmup_steps': 1, 'total_steps': 1, 'base_lr': 1e-3, 'step_num': 0}
    amp2 = A(); amp2.scaler = torch.cuda.amp.GradScaler(enabled=False)

    # Should restore weights + optimizer + scaler and return the saved step counter.
    step = load_checkpoint(m2, str(out/"model_last.pt"), optimizer=opt2, scheduler=sch2, amp=amp2)
    assert isinstance(step, int)