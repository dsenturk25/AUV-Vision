import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import random


def investigate_data(path: str):
    files = []
    for root, dirs, files in os.walk(path):
        print(root)
        print(dirs)
        print(len(files))
        print("--------------------------------")

    # train_data = random.sample(range(len(files)), k=(len(files) - 44))  # type: ignore
    # test_data = random.sample(range(len(files)), k=44)  # type: ignore


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
):
    train_data = datasets.ImageFolder(
        root=train_dir, transform=transform, target_transform=None
    )

    test_data = datasets.ImageFolder(
        root=test_dir, transform=transform, target_transform=None
    )

    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
