import torch
from torch.nn import Module

class LinearWarmupLR(Module):
    """
    Linear learning rate that ramps up from 0 to some lr over the course of the warmup  steps, then linearly adjusts to the final lr over a given number of steps.
    """
    def __init__(
            self, 
            optimizer,
            lr_after_warmup=1e-4,
            final_lr=1e-5,
            warmup_steps=10000,
            total_scheduled_steps=1000000
        ):
        super().__init__()
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        lr_warmup = torch.linspace(0.0, lr_after_warmup, warmup_steps)
        lr_after_warmup = torch.linspace(lr_after_warmup, final_lr, total_scheduled_steps - warmup_steps)
        self.register_buffer('lrs', torch.cat([lr_warmup, lr_after_warmup]))
        self.register_buffer('step', torch.tensor(-1))
        self.lr = self.lrs[0]

    def get_lr(self):
        return self.lrs[min(self.step, len(self.lrs) - 1)]
    
    def update(self):
        self.step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()