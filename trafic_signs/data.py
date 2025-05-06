import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    val_size: float,
) -> tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders.

    Args:
        data_dir (Path): Directory containing the dataset.
        batch_size (int): Batch size for DataLoader.
        val_size (float): Fraction of data to use for validation.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    NUM_WORKERS = os.cpu_count()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.3185, 0.2930, 0.3016), (0.2266, 0.2214, 0.2268)),
        ]
    )

    dataset = ImageFolder(data_dir, transform=transform)
    val_length = int(len(dataset) * val_size)
    train_length = len(dataset) - val_length

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_length, val_length],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_dataloader, val_dataloader


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """
    Denormalize the image tensor.

    Args:
        img (torch.Tensor): Image tensor to denormalize.

    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    mean = torch.tensor([0.3185, 0.2930, 0.3016])[:, None, None]
    std = torch.tensor([0.2266, 0.2214, 0.2268])[:, None, None]
    return img * std + mean
