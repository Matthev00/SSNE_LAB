import csv
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from data_utils import (
    TestDataset,
    get_data_loaders_equal_distribution,
    test_pad_collate,
)
from model import LSTMClassifier
from torch.utils.data import DataLoader

import wandb


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="compositor_classification")

    INPUT_SIZE = 1
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    OUTPUT_SIZE = 5
    LR = 0.0001
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    STEP_SIZE = 0.5
    GAMMA = 0.1
    DROPOUT = 0.1
    WEIGHT_DECAY = 0.05
    BIDIRECTIONAL = True

    CLASS_NAMES = ["bach", "beethoven", "debussy", "scarlatti", "victoria"]

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(device)

    train_dataloader, val_dataloader, class_weights_tensor = (
        get_data_loaders_equal_distribution(BATCH_SIZE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    from engine import train

    train(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        device=device,
        class_names=CLASS_NAMES,
    )

    with open("compositor_classification/data/test_no_target.pkl", "rb") as f:
        test_data = pickle.load(f)

    with open("compositor_classification/data/normalizer.pkl", "rb") as f:
        scaler = pickle.load(f)

    norm_test_data = [
        scaler.transform(torch.tensor(seq, dtype=torch.float32).reshape(-1, 1))
        .squeeze(1)
        .tolist()
        for seq in test_data
    ]

    test_dataset = TestDataset(norm_test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_pad_collate
    )

    model.eval()
    predictions = []
    with torch.inference_mode():
        for inputs in test_loader:
            inputs = inputs.to(device).unsqueeze(-1)
            logits = model(inputs)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            predictions.extend(preds.cpu().numpy())

    with open("pred.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for pred in predictions:
            writer.writerow([pred])


if __name__ == "__main__":
    main()
