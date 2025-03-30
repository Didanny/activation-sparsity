import torch
from torch import nn
from torchmetrics.aggregation import MeanMetric

class ActivationSparsity:
    def __init__(self):
        self.sparsity = {}
    
    def calculate_sparsity(self, name: str):
        
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor, name = name):
            mean_sparsity = (output.detach() == 0).float().mean().item()
            self._add_entry(name, mean_sparsity)
        
        return hook
    
    def _add_entry(self, key: str, val: float):
        if key in self.sparsity:
            self.sparsity[key].update(val)
        else:
            self.sparsity[key] = MeanMetric()
            self.sparsity[key].update(val)
            
    def reset(self):
        for key in self.sparsity:
            self.sparsity[key].reset() 
        
    def reinitialize(self):
        self.sparsity = {}
        
    def compile(self):
        return {key: val.compute().item() for key, val in self.sparsity.items()}
    
    def compute_average(self):
        return torch.mean(torch.tensor([val for key, val in self.compile().items() if key.endswith('relu')])).item()
    

class ActivationHoyerNorm(nn.Module):
    def __init__(self):
        self.running_norm = None
    
    def norm(self):
        
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            square_hoyer = (output.norm(p=1, dim=(1, 2, 3)) / output.norm(p=2, dim=(1, 2, 3))) ** 2
            self._accumulate(square_hoyer.mean())
        
        return hook
    
    def _accumulate(self, norm: torch.Tensor):
        if self.running_norm:
            self.running_norm += norm.mean()
        else:
            self.running_norm = torch.zeros_like(norm)
            self.running_norm += norm.mean()
    
    def reset(self):
        self.running_norm = None   
        
    def compute(self):
        return self.running_norm
