import csv
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from data_utils import (
    EmbeddingTestDataset,
    get_data_loaders_embedding,
)
from torch.nn.utils.rnn import pad_sequence
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
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    OUTPUT_SIZE = 5
    LR = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    STEP_SIZE = 5
    GAMMA = 0.1
    DROPOUT = 0.1
    WEIGHT_DECAY = 0.001
    EMBEDDING_DIM = 128
    VOCAB_SIZE = 182
    BIDIRECTIONAL = True

    CLASS_NAMES = ["bach", "beethoven", "debussy", "scarlatti", "victoria"]

    model = LSTMClassifier(
        embedding_dim=EMBEDDING_DIM,
        vocab_size=VOCAB_SIZE,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(device)

    train_dataloader, val_dataloader, class_weights_tensor = (
        get_data_loaders_embedding(BATCH_SIZE)
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
        
    with open("compositor_classification/data/vocab_mapping.pkl", "rb") as f:
        float2idx = pickle.load(f)

    test_dataset = EmbeddingTestDataset(test_data, float2idx)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=float2idx[-1.0])
    )

    model.eval()
    predictions = []
    with torch.inference_mode():
        for inputs in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            predictions.extend(preds.cpu().numpy())

    with open("pred.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for pred in predictions:
            writer.writerow([pred])


if __name__ == "__main__":
    main()
