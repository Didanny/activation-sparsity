from .utils import get_device, replace_gelu_with_relu
from .sparsity_utils import (
    ActivationHoyerNorm, 
    BlockHoyerNorm, 
    ActivationSparsity, 
    ActivationInspector,
    HooksManager,
)