from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path, index_col=None)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["HallwayType", "SubwayStation"])

    df["is_central_heating"] = df["HeatingType"].apply(
        lambda x: 1 if x == "central_heating" else 0
    )
    df = df.drop(columns=["HeatingType"])

    df["is_self_management"] = df["AptManageType"].apply(
        lambda x: 1 if x == "self_management" else 0
    )
    df = df.drop(columns=["AptManageType"])

    time_to_bus_stop_mapping = {"0~5min": 0, "5min~10min": 1, "10min~15min": 2}
    time_to_subway_mapping = {
        "0-5min": 0,
        "5min~10min": 1,
        "10min~15min": 2,
        "15min~20min": 3,
        "no_bus_stop_nearby": 4,
    }
    df["TimeToBusStop"] = df["TimeToBusStop"].map(time_to_bus_stop_mapping)
    df["TimeToSubway"] = df["TimeToSubway"].map(time_to_subway_mapping)

    df = df.astype({col: "int" for col in df.select_dtypes("bool").columns})

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    def encode_value(x):
        if x <= 100000:
            return 0
        elif 100000 < x <= 250000:
            return 1
        else:
            return 2

    df["Target"] = df["SalePrice"].apply(encode_value)
    df = df.drop(columns=["SalePrice"])
    return df


def prepare_data(data_path: Path, classification: bool = True) -> pd.DataFrame:
    df = load_data(data_path)
    df = encode_categorical_features(df)
    if classification:
        df = encode_target(df)
    else:
        df = df.rename(columns={"SalePrice": "Target"})

    return df


def compute_class_weights(y: pd.Series) -> dict:
    y.value_counts()
    class_counts = y.value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    return class_weights


def split_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


class HouseDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series = None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(
            self.y.iloc[idx], dtype=torch.long
        )


def create_data_loaders(
    data_path: Path,
    batch_size: int = 32,
    classification: bool = True,
    val_size: float = 0.2,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    df = prepare_data(data_path, classification)

    class_weights = torch.tensor([0])

    if classification:
        class_weights = compute_class_weights(df["Target"])

    X_train, X_val, y_train, y_val = split_data(df, val_size)

    train_dataset = HouseDataset(X_train, y_train)
    val_dataset = HouseDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_weights
