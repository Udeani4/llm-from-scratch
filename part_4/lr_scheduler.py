import math

class WarmupCosineLR:
    """Linear warmup → cosine decay (per-step API)."""
    # The standard LLM learning-rate schedule:
    #  - Warmup: ramp LR from ~0 up to base_lr. At the start, weights are random and Adam's
    #    running statistics are empty, so big steps can destabilise training.
    #  - Cosine decay: smoothly lower LR to 0 along half a cosine wave, letting the model
    #    settle into a good minimum at the end of training.
    #
    #   lr ^      ____
    #      |     /    `--.__
    #      |    /           `-.__
    #      |   /                 `--.
    #      +--+-------------------------> step
    #        warmup
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps+1, total_steps)
        self.base_lr = base_lr   # peak LR (reached at end of warmup)
        self.step_num = 0
    def step(self):
        # Call once per optimizer step; computes and applies the new LR.
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            # Linear ramp: base_lr * (step / warmup_steps)
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            # progress goes 0 -> 1 over the decay phase; cos(pi*progress) goes 1 -> -1,
            # so lr goes base_lr -> 0.
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        # Optimizers store LR per parameter group; overwrite it directly.
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr