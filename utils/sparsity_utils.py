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
            if len(output.shape) == 4: # Conv2D
                square_hoyer = (output.norm(p=1, dim=(1, 2, 3)) / output.norm(p=2, dim=(1, 2, 3))) ** 2
            elif len(output.shape) == 2: # Linear
                square_hoyer = (output.norm(p=1, dim=(1)) / output.norm(p=2, dim=(1))) ** 2
            elif len(output.shape) == 3: # MLP after Attention
                square_hoyer = (output.norm(p=1, dim=(1, 2)) / output.norm(p=2, dim=(1, 2))) ** 2
            else:
                raise ValueError
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


class HooksManager():
    def __init__(self, model: nn.Module, model_name: str):
        self.hooks = []
        # self.monitors = {}
        self.model = model
        self.model_name = model_name
        
    # def register_hooks(self, mode: str):
    #     assert mode == 'sparsity' or mode == 'hoyer'
            
    #     if mode == 'sparsity':
    #         self._register_sparsity_hooks()
        
    # def _register_sparsity_hooks(self, ):
    #     pass
        
    def register_hooks(self, hook_generator: callable, layers: str = 'all'):
        assert layers == 'all' or layers == 'post-act'
        
        hooks = []
        if layers == 'all':
            for name, mod in self.model.named_modules():
                name = self._standardize_name(name, mod)
                hooks.append(mod.register_forward_hook(hook_generator(name)))
                
        elif layers == 'post-act':
            
            if 'resnet' in self.model_name:
                for name, mod in self.model.named_modules():
                    if name.endswith('relu'):
                        hooks.append(mod.register_forward_hook(hook_generator()))
                        
            if 'vgg' in self.model_name:
                for name, mod in self.model.named_modules():
                    if isinstance(mod, nn.ReLU):
                        hooks.append(mod.register_forward_hook(hook_generator()))
                        
            if 'vit' in self.model_name:
                for name, mod in self.model.named_modules():
                    if isinstance(mod, nn.ReLU):
                        hooks.append(mod.register_forward_hook(hook_generator()))
                        
        self.hooks.append(hooks)
                        
    def remove_hooks(self):
        for hooks in self.hooks:
            for hook in hooks:
                hook.remove()
                
    def _standardize_name(self, name: str, mod: nn.Module):
        if isinstance(mod, nn.ReLU) and not name.endswith('relu'):
            return f'{name}_relu'
        elif isinstance(mod, nn.GELU) and not name.endswith('gelu'):
            return f'{name}_gelu'
        else:
            return name
                        