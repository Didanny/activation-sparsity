import torch
from torch import nn

def get_device():
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    return device

def replace_gelu_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, nn.ReLU())
        else:
            replace_gelu_with_relu(child)
            