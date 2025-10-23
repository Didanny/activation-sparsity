from pathlib import Path

from typing import Optional, Sequence, Union, TypedDict
from typing_extensions import Literal, TypeAlias

import torch.utils.data as data

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

DATA_DIR = Path('../datasets/')


def get_train_transforms(
    mean: Sequence[float], 
    std: list[float], 
    image_size: int = 32,
):
    transforms = [
        T.Resize(image_size),
        T.RandomCrop(image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]
    return T.Compose(transforms)

def get_val_transforms(
    mean: Sequence[float], 
    std: list[float], 
    image_size: int = 32,
):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def _cifar(
    root: Path, 
    image_size: tuple[int], 
    mean: list[float], 
    std: list[float], 
    batch_size: int, 
    num_workers: int, 
    dataset_builder: data.DataLoader, 
    **kwargs,
) -> tuple[data.DataLoader]:
    train_transforms = get_train_transforms(mean, std, image_size)
    val_transforms = get_val_transforms(mean, std, image_size)

    trainset = dataset_builder(root, train=True, transform=train_transforms, download=True)
    valset = dataset_builder(root, train=False, transform=val_transforms, download=True)

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

def cifar10(
    batch_size: Optional[int] = 256,
    image_size: Optional[int] = 32,
):
    return _cifar(
        root=DATA_DIR,
        image_size=image_size,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
        batch_size=batch_size,
        num_workers=2,
        dataset_builder=CIFAR10,
    )

def cifar100(
    batch_size: Optional[int] = 256,
    image_size: Optional[int] = 32,
):
    return _cifar(
        root=DATA_DIR,
        image_size=image_size,
        mean=[0.5070, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2761],
        batch_size=batch_size,
        num_workers=2,
        dataset_builder=CIFAR100,
    )
