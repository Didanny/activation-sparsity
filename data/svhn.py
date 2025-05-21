from pathlib import Path

from typing import Optional, Sequence, Union, TypedDict
from typing_extensions import Literal, TypeAlias

import torch.utils.data as data

import torchvision.transforms as T
from torchvision.datasets import SVHN

DATA_DIR = Path('../datasets/')

def get_train_transforms(mean: Sequence[float], std: list[float]):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def get_val_transforms(mean: Sequence[float], std: list[float]):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
def _svhn(root: Path, image_size: tuple[int], mean: list[float], std: list[float], batch_size: int, num_workers: int, dataset_builder: data.DataLoader, **kwargs) \
    -> tuple[data.DataLoader]:
    train_transforms = get_train_transforms(mean, std)
    val_transforms = get_val_transforms(mean, std)

    trainset = dataset_builder(root, split='train', transform=train_transforms, download=True)
    valset = dataset_builder(root, split='test', transform=val_transforms, download=True)

    # TODO: Maybe add support for distributed runs
    train_sampler = None
    val_sampler = None

    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=(train_sampler is None),
                                   sampler=train_sampler,
                                   num_workers=num_workers,
                                   persistent_workers=True)
    val_loader = data.DataLoader(valset, batch_size=batch_size,
                                 shuffle=False,
                                 sampler=val_sampler,
                                 num_workers=num_workers,
                                 persistent_workers=True)

    return train_loader, val_loader

def svhn(batch_size: Optional[int] = 256):
    return _svhn(
        root=DATA_DIR,
        image_size=32,
        mean=[0.4377, 0.4438, 0.4728],
        std=[0.1980, 0.2010, 0.1970],
        batch_size=batch_size,
        num_workers=2,
        dataset_builder=SVHN,
    )
