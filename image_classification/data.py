import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_balanced_subset(dataset: Dataset, data_percent: float) -> list[int]:
    """
    Get a balanced subset of the dataset.

    Args:
        dataset: (Dataset) The dataset to sample from.
        data_percent (float): Percentage of data to use.

    Returns:
        List of indices forming a balanced subset of the dataset.
    """
    assert 0.0 < data_percent <= 1.0, "Percentage must be in range (0.0, 1.0]"

    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)

    total_samples = int(len(dataset) * data_percent)
    num_classes = len(class_indices)
    n_per_class = total_samples // num_classes

    balanced_indices = []

    for label, indices in class_indices.items():
        available = len(indices)
        n_select = min(n_per_class, available)
        if n_select < n_per_class:
            print(
                f"⚠️ Warning: Class {label} has only {available} samples, selecting {n_select} instead of {n_per_class}."
            )
        random.shuffle(indices)
        balanced_indices.extend(indices[:n_select])

    print(
        f"✅ Final subset has {len(balanced_indices)} samples across {len(class_indices)} classes."
    )
    return balanced_indices


def split_dataset(indices: list[int], val_split: float) -> tuple[list[int], list[int]]:
    """
    Split list of indices into training and validation subsets.

    Args:
        indices: List of dataset indices.
        val_split: Fraction of data to use for validation.

    Returns:
        Tuple: (train_indices, val_indices)
    """
    assert 0.0 < val_split < 1.0, "Validation split must be in range (0.0, 1.0)"
    random.shuffle(indices)
    split = int(len(indices) * (1 - val_split))
    return indices[:split], indices[split:]


def get_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: transforms.Compose = None,
    val_transform: transforms.Compose = None,
    data_percent: float = 0.2,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """
    Load data and return train and validation data loaders with separate transforms.

    Args:
        data_dir (Path): Path to dataset folder.
        batch_size (int): Batch size.
        num_workers (int): Num workers.
        train_transform: Transform for training images.
        val_transform: Transform for validation images.
        data_percent (float): % of dataset to use.
        val_split (float): % of data to use for validation.

    Returns:
        Tuple of train and val DataLoaders.
    """
    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(24),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5204, 0.4950, 0.4381), (0.2113, 0.2103, 0.2100)
                ),
            ]
        )
    if val_transform is None:
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5204, 0.4950, 0.4381), (0.2113, 0.2103, 0.2100)
                ),
            ]
        )

    base_dataset = ImageFolder(data_dir)

    if data_percent < 1.0:
        selected_indices = get_balanced_subset(base_dataset, data_percent)
    else:
        selected_indices = list(range(len(base_dataset)))

    train_indices, val_indices = split_dataset(selected_indices, val_split)

    train_dataset = Subset(
        ImageFolder(data_dir, transform=train_transform), train_indices
    )
    val_dataset = Subset(ImageFolder(data_dir, transform=val_transform), val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def plot_class_distribution(subset: Subset):
    base_dataset = subset.dataset
    indices = subset.indices
    labels = [base_dataset.targets[i] for i in indices]

    class_counts = Counter(labels)

    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
    plt.xlabel("Class Index")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution in Subset")
    plt.xticks(range(len(class_counts)), rotation=90)
    plt.tight_layout()
    plt.show()


def main():
    data_dir = Path("image_classification/data/train")
    train_loader, val_loader = get_data_loaders(data_dir, data_percent=1)
    print(f"Train Loader: {len(train_loader.dataset)} samples")
    plot_class_distribution(train_loader.dataset)
    print(f"Validation Loader: {len(val_loader.dataset)} samples")
    plot_class_distribution(val_loader.dataset)


if __name__ == "__main__":
    main()
