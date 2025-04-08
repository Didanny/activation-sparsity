import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric

from utils import ActivationSparsity, ActivationHoyerNorm, HooksManager, get_device
import models
import data

from torch.utils.tensorboard import SummaryWriter

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cifar100_resnet20')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--fine-tune', action='store_true')
    parser.add_argument('--alpha', type=float, default=10e-7)
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
    
def log_meters(writer: SummaryWriter, meters: dict, prefix: str, step: int):    
    # Log meters to Tensorboard
    writer.add_scalar(f'{prefix}/loss', meters['loss'].compute(), step)
    for k in [1, 5]:
        writer.add_scalar(f'{prefix}/top{k}_accuracy', meters[f'top{k}_accuracy'].compute(), step)
            
    # Reset meters
    for m in meters.values():
        m.reset()
        
def log_sparsity(writer: SummaryWriter, sparsities: dict, prefix: str, step: int):
    # Log sparsity values to Tensorboard
    for key, val in sparsities.items():
        writer.add_scalar(f'{prefix}/{key}', val, step)

def train(model: nn.Module, criterion: nn.Module, optimizer: nn.Module, scheduler: object, train_loader: DataLoader, 
          device: torch.device, epoch: int, meters: dict) -> None:
    # Training mode
    model.train()
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}'), 0):
        # Load training data sample and label
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Get loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
        # Backprop
        loss.backward()
        
        # Update params
        optimizer.step()
        
        # Update the metrics
        if meters is not None:
            # Update accuracies
            for k in [1, 5]:
                meters[f'top{k}_accuracy'].update(outputs, labels)
            # Update running loss    
            meters['loss'].update(loss)
    
    # Update learning rate
    if scheduler:
        scheduler.step()
            
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
            for k in [1, 5]:
                meters[f'top{k}_accuracy'].update(outputs, labels)
            # Update running loss    
            meters['loss'].update(loss)

def main(opt: argparse.Namespace):
    # Get the current device
    device = get_device()
    
    # Set up tensorboard summary writer
    # TODO: Create more comprehensive automated commenting
    if not opt.fine_tune:
        writer = SummaryWriter(comment=f'_{opt.model}_{opt.dataset}')
    else:
        writer = SummaryWriter()
        log_dir = writer.log_dir.replace('runs', 'fine_tune_runs', 1)
        writer = SummaryWriter(log_dir=f'{log_dir}_{opt.model}_{opt.dataset}_{opt.alpha}')
    save_dir = Path(writer.log_dir)
    
    # Directories
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    train_loader, val_loader = getattr(data, opt.dataset)()
    
    # Model
    model = getattr(models, opt.model)(pretrained = opt.pretrained, num_classes = data.num_classes[opt.dataset])
    model.to(device = device)
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = optim.SGD([v for n, v in model.named_parameters()], 0.1, 0.9, 0, 5e-4, True)
    
    # Initialize scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    
    # Get the performance metrics
    train_meters = get_meters(device, 'train', data.num_classes[opt.dataset])
    val_meters = get_meters(device, 'val', data.num_classes[opt.dataset])
    
    # Initialize best and last model metrics
    best_dict, last_dict, best_fitness = None, None, 0.0
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Begin training
    if not opt.pretrained:
        for epoch in range(opt.epochs):
            # Train
            train(model, criterion, optimizer, lr_scheduler, train_loader, device, epoch, train_meters)
            log_meters(writer, train_meters, 'train', epoch)
            
            # Eval
            if (epoch + 1) % 5 == 0:
                evaluate(model, criterion, val_loader, device, epoch, val_meters)
            
                # Get fitness
                fitness = val_meters['top1_accuracy'].compute()
                top5_accuracy, top1_accuracy = val_meters['top5_accuracy'].compute(), val_meters['top1_accuracy'].compute() 

                # Update best model
                if best_fitness < fitness:
                    best_dict = {'params': model.state_dict(), 'top5_accuracy': top5_accuracy, 'top1_accuracy': top1_accuracy, 'epoch': epoch}
                    best_fitness = fitness
                    torch.save(best_dict, best)
                    
                # Log val meters
                log_meters(writer, val_meters, 'val', epoch) 
                
        # Update last model
        last_dict = {'params': model.state_dict(), 'top5_accuracy': top5_accuracy, 'top1_accuracy': top1_accuracy, 'epoch': epoch}
        torch.save(last_dict, last)
        
    # Initialise hooks manager
    hook_manager = HooksManager(model, opt.model)
    
    # Measure initial sparsity on validation set
    natural_sparsity = ActivationSparsity()
    sparsity_vals = {}
    
    hook_manager.register_hooks(natural_sparsity.calculate_sparsity)
        
    # Run on validation set
    evaluate(model, criterion, val_loader, device, 0, val_meters)
    sparsity_vals['natural'] = natural_sparsity.compute_average()
    
    hook_manager.remove_hooks()
    
    # Induce activation sparsity with Hoyer Regularizer
    induced_sparsity = ActivationSparsity()
    hoyer_norm = ActivationHoyerNorm()

    hook_manager.register_hooks(hoyer_norm.norm, layers='post-relu')
    hook_manager.register_hooks(induced_sparsity.calculate_sparsity)
            
    # Initialize criterion
    loss = nn.CrossEntropyLoss()
    def criterion(outputs: torch.Tensor, labels: torch.Tensor):
        result = loss(outputs, labels) + (opt.alpha * hoyer_norm.compute())
        hoyer_norm.reset()
        return result
    
    # Initialize optimizer
    optimizer = optim.SGD([v for n, v in model.named_parameters()], 0.005, 0.9, 0, 5e-4, True)
    
    # Initialize scheduler
    # lr_scheduler = None
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.finetune_epochs, eta_min=0)
    
    # Get the performance metrics
    train_meters = get_meters(device, 'train', data.num_classes[opt.dataset])
    val_meters = get_meters(device, 'val', data.num_classes[opt.dataset]) 
    
    # Initialize best and last model metrics
    best_dict, last_dict, best_fitness = None, None, 0.0
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Begin fine-tuning with Hoyer activation regularizer
    if opt.fine_tune:
        for epoch in range(opt.finetune_epochs):
            # Train
            train(model, criterion, optimizer, lr_scheduler, train_loader, device, epoch, train_meters)
            
            # Compute induced sparsity
            sparsity_vals['induced_sparsity'] = induced_sparsity.compute_average()
            induced_sparsity.reset()
            
            # Log to Tensorboard
            log_meters(writer, train_meters, 'train', epoch)
            log_sparsity(writer, sparsity_vals, 'train', epoch)
            
            # Eval
            evaluate(model, criterion, val_loader, device, epoch, val_meters)
            
            # Compute induced sparsity
            sparsity_vals['induced_sparsity'] = induced_sparsity.compute_average()
            induced_sparsity.reset()
            
            # Log to Tensorboard
            log_meters(writer, val_meters, 'val', epoch)
            log_sparsity(writer, sparsity_vals, 'val', epoch)
        
    hook_manager.remove_hooks()
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    