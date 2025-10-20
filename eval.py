import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric

from utils import (
    ActivationSparsity, 
    ActivationHoyerNorm, 
    BlockHoyerNorm, 
    HooksManager, 
    SparseActivationEnforcer,
    get_device, 
    replace_gelu_with_relu,
    get_lr_scheduler,
    get_optimizer,
)
import models
import data

from torch.utils.tensorboard import SummaryWriter

checkpoints = {
    'cifar10_hoyer': 'fine_tune_runs/May29_08-57-52_korn.ics.uci.edu_cifar10_vit_cifar10_1e-05_unstructured/weights/last.pt',
    'cifar10_block_hoyer': 'fine_tune_runs/May29_03-45-52_korn.ics.uci.edu_cifar10_vit_cifar10_0.05_semi-structured/weights/last.pt',
    'cifar100_hoyer': 'fine_tune_runs/May02_18-38-11_korn.ics.uci.edu_cifar100_vit_cifar100_5e-06/weights/last.pt',
    'cifar100_block_hoyer': 'fine_tune_runs/May04_03-08-28_korn.ics.uci.edu_cifar100_vit_cifar100_0.1_semi-structured/weights/last.pt',
    'tinyimagenet_hoyer': 'fine_tune_runs/May30_01-28-32_korn.ics.uci.edu_tinyimagenet_vit_tinyimagenet_5e-06_unstructured/weights/last.pt',
    'tinyimagenet_block_hoyer': '/home/dannya1/activation-sparsity/fine_tune_runs/May29_17-24-00_korn.ics.uci.edu_tinyimagenet_vit_tinyimagenet_0.1_semi-structured/weights/last.pt',
}

layers = [
    'enc.2.mlp.4_relu',
    'enc.3.mlp.1_relu',
    'enc.4.mlp.1_relu',
    'enc.5.mlp.1_relu',
    'enc.4.mlp.4_relu',
    'enc.2.mlp.1_relu',
    'enc.3.mlp.4_relu',
    'enc.5.mlp.4_relu',
    'enc.6.mlp.4_relu',
    'enc.1.mlp.1_relu',
    'enc.6.mlp.1_relu',
    'enc.0.mlp.4_relu',
    'enc.0.mlp.1_relu',
    'enc.1.mlp.4_relu',
]

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cifar10_vit')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--checkpoint', type=str, default='cifar10_hoyer')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def get_meters(device: torch.device, phase: str, num_classes: int):
    """util function for meters"""
    def get_single_meter(phase):
        meters = {}
        meters['loss'] = MeanMetric()
        meters['loss'] = meters['loss'].to(device = device)
        for k in [1, 5]:
            meters[f'top{k}_accuracy'] = Accuracy(task='multiclass', num_classes= num_classes, top_k=k)
            meters[f'top{k}_accuracy'] = meters[f'top{k}_accuracy'].to(device = device)
        return meters

    assert phase in ['train', 'val', 'test'], 'Invalid phase.'
    return get_single_meter(phase)
           
@torch.no_grad()    
def evaluate(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int, meters: dict):
    # Eval mode
    model.eval()
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}'), 0):
        # Load validation data sample and label
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
            
        # Get loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Update the metrics
        if meters is not None:
            # Update accuracies
            meters[f'top1_accuracy'].update(outputs, labels)
            # Update running loss    
            meters['loss'].update(loss)

def main(opt: argparse.Namespace):
    # Get the current device
    device = get_device()
    
    # Dataset
    train_loader, val_loader = getattr(data, opt.dataset)(batch_size = 1)
    
    # Model
    model = getattr(models, opt.model)(num_classes = data.num_classes[opt.dataset])
    replace_gelu_with_relu(model)
    checkpoint = torch.load(checkpoints[opt.checkpoint], map_location='cpu')
    model.load_state_dict(checkpoint['params'])
    model.to(device = device)
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Get the performance metrics
    pre_enforce_meters = get_meters(device, 'val', data.num_classes[opt.dataset])
    post_enforce_meters = get_meters(device, 'val', data.num_classes[opt.dataset])
    
    # Initial evaluation
    evaluate(model, criterion, val_loader, device, 0, pre_enforce_meters)
    print(f'Pre-enforcement accuracy: {pre_enforce_meters["top1_accuracy"].compute()}')
    
    # Initialise hooks manager
    hook_manager = HooksManager(model, opt.model)
    
    # Add the 2:4 enforcer
    enforcer = SparseActivationEnforcer()
    hook_manager.register_hooks(enforcer.enforce, layers)
    
    # Final evaluation
    evaluate(model, criterion, val_loader, device, 0, post_enforce_meters)
    print(f'Pre-enforcement accuracy: {post_enforce_meters["top1_accuracy"].compute()}')
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    