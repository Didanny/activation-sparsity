import argparse
import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

def get_lr_scheduler(optimizer: torch.optim.Optimizer, opt: argparse.Namespace):
    if opt.lr_scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.final_lr)
    elif opt.lr_scheduler == 'cosine-warmup':
        return CosineAnnealingWithWarmup(optimizer, total_epochs=opt.epochs, warmup_epochs=opt.warmup_epochs, 
                                         initial_lr=opt.initial_lr, final_lr=opt.final_lr)

class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs, initial_lr, final_lr, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = torch.tensor(self.last_epoch + 1, dtype=torch.float32)

        if epoch < self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            return [self.initial_lr * warmup_factor.item() for _ in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + torch.cos(torch.pi * progress))
            return [(self.final_lr + (self.initial_lr - self.final_lr) * cosine_factor).item()
                    for _ in self.base_lrs]
