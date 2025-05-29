from .cifar import cifar10, cifar100
from .svhn import svhn
from .tinyimagenet import tinyimagenet_hf as tinyimagenet

num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
    'tinyimagenet': 200,
}
