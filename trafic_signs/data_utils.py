import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    subset_size: int = None,
    val_size: int = 1000,
) -> tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders.

    Args:
        data_dir (Path): Directory containing the dataset.
        batch_size (int): Batch size for DataLoader.
        subset_size (int, optional): Number of samples to use for training and validation. Defaults to None.


    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    NUM_WORKERS = os.cpu_count()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageFolder(data_dir, transform=transform)
    if subset_size is not None:
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    train_length = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_length, val_size],
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
    mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    return img * std + mean


def export_real_images_for_fid(
    val_loader: torch.utils.data.DataLoader,
    output_dir: Path,
) -> None:
    """
    Exports denormalized real images from val_loader as .png for FID calculation.

    Args:
        val_loader (DataLoader): Validation DataLoader.
        output_dir (Path): Path to save images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    to_pil = transforms.ToPILImage()

    for images, _ in val_loader:
        for img in images:
            img = denormalize(img).clamp(0, 1)
            pil_img = to_pil(img)
            pil_img.save(output_dir / f"{saved:04}.png")
            saved += 1


def build_class_distribution() -> dict[int, float]:
    """
    Build class distribution for FID calculation.
    """
    class_counts = {
        0: 210,
        1: 2220,
        2: 2250,
        3: 1410,
        4: 1980,
        5: 1860,
        6: 420,
        7: 1440,
        8: 1410,
        9: 1470,
        10: 2010,
        11: 1320,
        12: 2100,
        13: 2160,
        14: 780,
        15: 630,
        16: 420,
        17: 1110,
        18: 1200,
        19: 210,
        20: 360,
        21: 330,
        22: 390,
        23: 510,
        24: 270,
        25: 1500,
        26: 600,
        27: 240,
        28: 540,
        29: 270,
        30: 450,
        31: 780,
        32: 240,
        33: 689,
        34: 420,
        35: 1200,
        36: 390,
        37: 210,
        38: 2070,
        39: 300,
        40: 360,
        41: 240,
        42: 240,
    }
    total = sum(class_counts.values())
    return {cls: count / total for cls, count in class_counts.items()}
