import torch

import wandb
from data_preparation import create_data_loaders
from engine import train
from model import HouseNet, HybridNet


def main():
    torch.manual_seed(42)

    wandb.init()
    config = wandb.config

    MODEL_TYPE = config.MODEL_TYPE
    BATCH_NORM = config.BATCH_NORM

    EPOCHS = config.epochs
    HIDDEN_DIM = config.HIDDEN_DIM
    NUM_LAYERS = config.NUM_LAYERS
    DROPOUT = config.DROPOUT
    BATCH_SIZE = config.BATCH_SIZE
    LR = config.LR
    WEIGHT_DECAY = config.WEIGHT_DECAY

    DATA_PATH = "train_data.csv"
    INPUT_SIZE = 27
    VAL_SIZE = 0.2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, val_dataloader, class_weights = create_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
    )

    if MODEL_TYPE == "classifier":
        OUTPUT_SIZE = 3
        model = HouseNet(
            input_size=INPUT_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_size=OUTPUT_SIZE,
            dropout=DROPOUT,
            batch_norm=BATCH_NORM,
        ).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

        def train_fn():
            train(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=EPOCHS,
                device=device,
                model_type=MODEL_TYPE,
            )

    elif MODEL_TYPE == "regressor":
        OUTPUT_SIZE = 1
        model = HouseNet(
            input_size=INPUT_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_size=OUTPUT_SIZE,
            dropout=DROPOUT,
            batch_norm=BATCH_NORM,
        ).to(device)
        loss_fn = torch.nn.MSELoss().to(device)

        def train_fn():
            train(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=EPOCHS,
                device=device,
                model_type=MODEL_TYPE,
            )

    elif MODEL_TYPE == "hybrid":
        model = HybridNet(
            input_size=INPUT_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_norm=BATCH_NORM,
        ).to(device)
        loss_fn_reg = torch.nn.MSELoss().to(device)
        loss_fn_class = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

        def train_fn():
            train(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                loss_fn=(loss_fn_reg, loss_fn_class),
                epochs=EPOCHS,
                device=device,
                model_type=MODEL_TYPE,
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_fn()

    wandb.finish()
