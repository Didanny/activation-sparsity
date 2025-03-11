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
    