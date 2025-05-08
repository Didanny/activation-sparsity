import torch
import torch.nn as nn

from models import cifar100_vit
from data import cifar100
from utils import ActivationSparsity, ActivationInspector, HooksManager, get_device, replace_gelu_with_relu
from train import evaluate

dense_model = cifar100_vit(pretrained=False)
replace_gelu_with_relu(dense_model)

checkpoint_path = './fine_tune_runs/May02_18-38-11_korn.ics.uci.edu_cifar100_vit_cifar100_5e-06/weights/last.pt'
unstructured_checkpoint = torch.load(checkpoint_path, map_location='cpu')
unstructured_model = cifar100_vit(pretrained=False)
unstructured_model.load_state_dict(unstructured_checkpoint['params'])
replace_gelu_with_relu(unstructured_model)

checkpoint_path = './fine_tune_runs/May04_03-08-28_korn.ics.uci.edu_cifar100_vit_cifar100_0.1_semi-structured/weights/last.pt'
semi_checkpoint = torch.load(checkpoint_path, map_location='cpu')
semi_model = cifar100_vit(pretrained=False)
semi_model.load_state_dict(semi_checkpoint['params'])
replace_gelu_with_relu(semi_model)

dense_hooks = HooksManager(dense_model, 'cifar100_vit')
unstructured_hooks = HooksManager(unstructured_model, 'cifar100_vit')
semi_hooks = HooksManager(semi_model, 'cifar100_vit')

dense_sparsity = ActivationSparsity()
unstructured_sparsity = ActivationSparsity()
semi_sparsity = ActivationSparsity()

# unstructured_activations = ActivationInspector()
# semi_activations = ActivationInspector()

train_loader, val_loader = cifar100()

dense_hooks.register_hooks(dense_sparsity.calculate_sparsity)
unstructured_hooks.register_hooks(unstructured_sparsity.calculate_sparsity)
semi_hooks.register_hooks(semi_sparsity.calculate_sparsity)

# unstructured_hooks.register_hooks(unstructured_activations.store_activations)
# semi_hooks.register_hooks(semi_activations.store_activations)

evaluate(dense_model, nn.CrossEntropyLoss(), val_loader, 'cpu', 0, None)
evaluate(unstructured_model, nn.CrossEntropyLoss(), val_loader, 'cpu', 0, None)
evaluate(semi_model, nn.CrossEntropyLoss(), val_loader, 'cpu', 0, None)

print(semi_sparsity.compute_average(), unstructured_sparsity.compute_average())

torch.save({
    'dense_sparsity': dense_sparsity.compile(),
    'unstructured_sparsity': unstructured_sparsity.compile(),
    'semistructured_sparsity': semi_sparsity.compile(),
    # 'unstructured_activations': unstructured_activations.compile(),
    # 'semistructured_activations': semi_activations.compile(),
}, './results/sparsities_activations.pt')