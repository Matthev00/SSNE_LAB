import pickle
import random
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class VariableLenDataset(Dataset):
    def __init__(self, in_data, target):
        self.data = [(x, y) for x, y in zip(in_data, target)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data, target = self.data[idx]
        in_data = torch.tensor(in_data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)
        return in_data, target


class EmbeddingDataset(Dataset):
    def __init__(self, in_data: list[list[float]], targets: list[int], float2idx: dict[float, int]):
        self.data = [(x, y) for x, y in zip(in_data, targets)]
        self.float2idx = float2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data, target = self.data[idx]
        encoded = [self.float2idx.get(v, self.float2idx[-1.0]) for v in in_data]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class TestDataset(Dataset):
    def __init__(self, in_data):
        self.data = in_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data = self.data[idx]
        in_data = torch.tensor(in_data, dtype=torch.float32)
        return in_data


class EmbeddingTestDataset(Dataset):
    def __init__(self, in_data: list[list[float]], float2idx: dict[float, int]):
        self.data = in_data
        self.float2idx = float2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        encoded = [self.float2idx.get(v, self.float2idx[-1.0]) for v in seq]
        return torch.tensor(encoded, dtype=torch.long)


def test_pad_collate(batch, pad_value=0):
    xx = [
        torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        for x in batch
    ]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)

    return xx_pad


def pad_collate(batch, pad_value=0):
    xx, yy = zip(*batch)
    xx = [
        torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        for x in xx
    ]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.tensor(yy, dtype=torch.long)

    return xx_pad, yy


def pad_collate_embedding(batch, pad_value=0):
    xx, yy = zip(*batch)
    xx = [
        torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x
        for x in xx
    ]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.tensor(yy, dtype=torch.long)
    return xx_pad, yy


def get_data_loaders_embedding(batch_size=50):
    import pickle
    from collections import Counter
    from sklearn.model_selection import train_test_split

    with open("compositor_classification/data/train.pkl", "rb") as f:
        train_data = pickle.load(f)

    data = []
    targets = []
    for x, y in train_data:
        data.append(x)
        targets.append(y)

    unique_vals = sorted(set(val for seq in data for val in seq if val != -1.0))
    float2idx = {val: i for i, val in enumerate(unique_vals)}
    float2idx[-1.0] = len(float2idx)

    with open("compositor_classification/data/vocab_mapping.pkl", "wb") as f:
        pickle.dump(float2idx, f)

    X_train, X_val, y_train, y_val = train_test_split(
        data, targets, test_size=0.2, stratify=targets, random_state=42
    )

    train_set = EmbeddingDataset(X_train, y_train, float2idx)
    val_set = EmbeddingDataset(X_val, y_val, float2idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_embedding
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_embedding
    )

    class_counts = Counter(y_train)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    num_classes = len(class_weights)
    class_weights_tensor = torch.tensor(
        [class_weights[i] for i in range(num_classes)], dtype=torch.float32
    )

    return train_loader, val_loader, class_weights_tensor


def get_data_loaders(batch_size: int = 50):
    with open("compositor_classification/data/train.pkl", "rb") as f:
        train_data = pickle.load(f)

    data = []
    targets = []
    max_val = -1
    for sample in range(len(train_data)):
        data.append(train_data[sample][0])
        targets.append(train_data[sample][1])
        max_val = max(max_val, max(data[-1]))

    train_indices = int(len(train_data) * 0.8)
    data = [(x / max_val) for x in data]
    train_set = VariableLenDataset(data[:train_indices], targets[:train_indices])
    val_set = VariableLenDataset(data[train_indices:], targets[train_indices:])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate,
    )

    return train_loader, val_loader


def balance_classes_with_sampling(data, targets, fragment_length=1000):

    class_counts = Counter(targets)
    max_count = max(class_counts.values())

    balanced_data = list(data)
    balanced_targets = list(targets)

    for cls, count in class_counts.items():
        samples_to_generate = max_count - count

        class_samples = [data[i] for i in range(len(data)) if targets[i] == cls]

        for _ in range(samples_to_generate):
            sample = random.choice(class_samples)

            if len(sample) <= fragment_length:
                fragment = sample
            else:
                start_idx = random.randint(0, len(sample) - fragment_length)
                fragment = sample[start_idx : start_idx + fragment_length]

            balanced_data.append(fragment)
            balanced_targets.append(cls)

    return balanced_data, balanced_targets


def get_data_loaders_equal_distribution(batch_size=50):
    with open("compositor_classification/data/train.pkl", "rb") as f:
        train_data = pickle.load(f)

    data = []
    targets = []

    for x, y in train_data:
        data.append(x)
        targets.append(y)

    all_flat = torch.tensor(
        [item for seq in data for item in seq], dtype=torch.float32
    ).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(all_flat.numpy())

    with open("compositor_classification/data/normalizer.pkl", "wb") as f:
        pickle.dump(scaler, f)

    norm_data = [
        scaler.transform(torch.tensor(seq, dtype=torch.float32).reshape(-1, 1))
        .squeeze(1)
        .tolist()
        for seq in data
    ]

    X_train, X_val, y_train, y_val = train_test_split(
        norm_data, targets, test_size=0.2, stratify=targets, random_state=42
    )
    X_train, y_train = balance_classes_with_sampling(X_train, y_train)

    train_set = VariableLenDataset(X_train, y_train)
    val_set = VariableLenDataset(X_val, y_val)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate,
    )

    class_counts = Counter(y_train)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    num_classes = len(class_weights)
    class_weights_tensor = torch.tensor(
        [class_weights[i] for i in range(num_classes)], dtype=torch.float32
    )

    return train_loader, val_loader, class_weights_tensor


def get_class_distribution(dataloader, name="train"):
    all_labels = []
    for _, targets in dataloader:
        all_labels.extend(targets.tolist())

    label_counts = Counter(all_labels)
    print(f"\nRozkład klas w zbiorze {name}:")
    for label, count in sorted(label_counts.items()):
        print(f"  Klasa {label}: {count} próbek")


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()
    get_class_distribution(train_loader, "train")
    get_class_distribution(val_loader, "val")
    equal_train_loader, equal_val_loader, class_weight = (
        get_data_loaders_equal_distribution()
    )
    get_class_distribution(equal_train_loader, "train (equal distribution)")
    get_class_distribution(equal_val_loader, "val (equal distribution)")
