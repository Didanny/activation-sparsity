import argparse
import torch
from torch import nn
from torch import optim

def get_device():
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    return device

def get_optimizer(model: nn.Module, opt: argparse.Namespace):
    if opt.optimizer == 'sgd':
        return optim.SGD([v for n, v in model.named_parameters()], opt.initial_lr, 0.9, 0, 5e-4, True)
    elif opt.optimizer == 'adam':
        return optim.Adam([v for n, v in model.named_parameters()], opt.initial_lr, weight_decay=5e-5)

def replace_gelu_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, nn.ReLU())
        else:
            replace_gelu_with_relu(child)
            