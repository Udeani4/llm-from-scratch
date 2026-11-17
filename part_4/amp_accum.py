import torch

class AmpGrad:
    """AMP + gradient accumulation wrapper.
    Usage:
        amp = AmpGrad(optimizer, accum=4, amp=True)
        amp.backward(loss)
        if amp.should_step(): amp.step(); amp.zero_grad()
    """
    # Two memory/speed tricks for training bigger models on limited GPUs:
    #  - AMP (automatic mixed precision): run most math in fp16 -> ~2x faster, half the activation memory.
    #  - Gradient accumulation: run `accum` small micro-batches, summing their gradients, then do ONE
    #    optimizer step. Effective batch size = batch_size * accum, without needing the memory for it.
    def __init__(self, optimizer, accum: int = 1, amp: bool = True):
        self.optim = optimizer
        self.accum = max(1, accum)
        self.amp = amp and torch.cuda.is_available()   # fp16 AMP only makes sense on CUDA
        # Scales the loss up before backward so small fp16 gradients don't underflow to zero.
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self._n = 0   # number of micro-batches seen so far
    def backward(self, loss: torch.Tensor):
        # Divide by accum so the SUM of accumulated gradients equals the gradient of the AVERAGE loss
        # over the big effective batch (otherwise the effective learning rate would be accum times larger).
        loss = loss / self.accum
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        # Gradients ADD UP in .grad across backward() calls until zero_grad() — that's the accumulation.
        self._n += 1
    def should_step(self):
        # True on every accum-th micro-batch.
        return (self._n % self.accum) == 0
    def step(self):
        if self.amp:
            # Un-scales gradients, skips the update if any are inf/NaN, then adjusts the scale factor.
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
    def zero_grad(self):
        self.optim.zero_grad(set_to_none=True)