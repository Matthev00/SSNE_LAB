import random

import numpy as np
import torch
import torch.nn as nn
from data_utils import get_data_loaders_equal_distribution, get_data_loaders_embedding
from engine import train
from model import LSTMClassifier

import wandb


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="compositor_classification")

    config = wandb.config

    INPUT_SIZE = config.input_size
    HIDDEN_SIZE = config.hidden_size
    NUM_LAYERS = config.num_layers
    OUTPUT_SIZE = config.output_size
    LR = config.lr
    BATCH_SIZE = config.batch_size
    NUM_EPOCHS = config.num_epochs
    STEP_SIZE = config.step_size
    GAMMA = config.gamma
    DROPOUT = config.dropout
    WEIGHT_DECAY = config.weight_decay
    BIDIRECTIONAL = config.bidirectional
    EMBEDDING_DIM = config.embedding_dim

    CLASS_NAMES = ["bach", "beethoven", "debussy", "scarlatti", "victoria"]

    model = LSTMClassifier(
        vocab_size=182,
        embedding_dim=EMBEDDING_DIM,
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
    if config.scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=STEP_SIZE, gamma=GAMMA
        )
    elif config.scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=2
        )
    elif config.scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )

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


if __name__ == "__main__":
    main()
