import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


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
    

def pad_collate(batch, pad_value=0):
    xx, yy = zip(*batch)
    xx = [torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x for x in xx]
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.tensor(yy, dtype=torch.long)

    return xx_pad, yy, x_lens


def get_data_loaders(batch_size=50):
    with open('compositor_classification/data/train.pkl', 'rb') as f:
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
        train_set, batch_size=50, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=50, shuffle=False, drop_last=False, collate_fn=pad_collate
    )

    return train_loader, val_loader

