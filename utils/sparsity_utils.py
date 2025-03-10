import torch
from torchmetrics.aggregation import MeanMetric

class ActivationSparsity:
    def __init__(self):
        self.sparsity = {}
    
    def calculate_sparsity(self, name: str):
        pass
    
    def _add_entry(self, key: str, val: torch.Tensor):
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
    