import torch
from torch import nn
from torchmetrics.aggregation import MeanMetric
from torch.sparse import to_sparse_semi_structured

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
    
    
class ActivationInspector:
    def __init__(self):
        self.activations = {}
    
    def store_activations(self, name: str):
        
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor, name = name):
            activation = output.detach()
            activation = activation.to(device = 'cpu')
            self._add_entry(name, activation)
        
        return hook
    
    def _add_entry(self, key: str, val: torch.Tensor):
        if key in self.activations:
            self.activations[key].append(val)
        else:
            self.activations[key] = [val]
            
    def reset(self):
        for key in self.activations:
            self.activations[key] = []
        
    def reinitialize(self):
        self.activations = {}
        
    def compile(self):
        return {key: torch.cat(val, dim=0) for key, val in self.activations.items()}


class SparseActivationEnforcer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def enforce(self):
        
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            batch_size, seq_len, hidden_dim = output.shape

            assert batch_size == 1, "Batch size must be 1 for 2:4 sparsity enforcement"
            assert output.view(-1).size(0) % 4 == 0, "Dimension must be divisible by 4 for 2:4 sparsity."

            output_half = to_sparse_semi_structured(output[:, 1:, :].squeeze(0).half()).to_dense().unsqueeze(0)
            mask = output_half != 0
            sparse_output = output[:, 1:, :] * mask
            
            return torch.cat((output[:, 0:1, :], sparse_output), dim=1)
        
        return hook


class ActivationHoyerNorm(nn.Module):
    def __init__(self):
        self.running_norm = None
    
    def norm(self):
        
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            if len(output.shape) == 4: # Conv2D
                l1 = output.norm(p=1, dim=(1, 2, 3))
                l2 = output.norm(p=2, dim=(1, 2, 3))
            elif len(output.shape) == 2: # Linear
                l1 = output.norm(p=1, dim=1)
                l2 = output.norm(p=2, dim=1)
            elif len(output.shape) == 3: # MLP after Attention
                l1 = output.norm(p=1, dim=(1, 2))
                l2 = output.norm(p=2, dim=(1, 2))
            else:
                raise ValueError
            
            square_hoyer = (l1 / (l2 + 1e-6)) ** 2
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
    

class BlockHoyerNorm(nn.Module):
    def __init__(self):
        self.running_norm = None
    
    def norm(self):
        
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            if len(output.shape) == 4: # Conv2D
                raise NotImplementedError
            elif len(output.shape) == 2: # Linear
                raise NotImplementedError
            elif len(output.shape) == 3: # MLP after Attention
                batch_size, seq_len, hidden_dim = output.shape
                output_flat = output.view(batch_size, -1)  # Flatten to (batch_size, seq_len * hidden_dim)

                assert output_flat.size(1) % 4 == 0, "Dimension must be divisible by 4 for 2:4 sparsity."

                output_blocks = output_flat.view(batch_size, -1, 4)  # (batch_size, num_blocks, 4)

                l1_norm_blocks = output_blocks.norm(p=1, dim=2)
                l2_norm_blocks = output_blocks.norm(p=2, dim=2)

                block_hoyer = (l1_norm_blocks / (l2_norm_blocks + 1e-6)) ** 2  # Add epsilon for stability
            else:
                raise ValueError
            self._accumulate(block_hoyer.mean())
        
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
        
    def register_hooks(self, hook_generator: callable, layers: str | list = 'all'):
        assert layers == 'all' or layers == 'post-act' or type(layers) == list
        
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
                        
        else:
            for name, mod in self.model.named_modules():
                name = self._standardize_name(name, mod)
                if name in layers:
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
                        