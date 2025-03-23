import torch
import wandb

from data_preparation import create_data_loaders
from engine import train
from model import HouseNet


def main():
    torch.manual_seed(42)

    wandb.init()
    config = wandb.config

    CLS = True

    EPOCHS = config.epochs
    HIDDEN_DIM = config.HIDDEN_DIM
    NUM_LAYERS = config.NUM_LAYERS
    DROPOUT = config.DROPOUT
    BATCH_SIZE = config.BATCH_SIZE
    LR = config.LR
    WEIGHT_DECAY = config.WEIGHT_DECAY
    
    CONFUSION_MATRIX = True
    DATA_PATH = "train_data.csv"
    OUTPUT_SIZE = 3 if CLS else 1
    INPUT_SIZE = 27
    VAL_SIZE = 0.2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, val_dataloader, class_weights = create_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        classification=CLS,
        val_size=VAL_SIZE,
    )

    model = HouseNet(
        input_size=INPUT_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    wandb.init(
        project="House Cost",
        entity="michall00-warsaw-university-of-technology",
        config={
            "epochs": EPOCHS,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "classification": CLS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
    )

    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=EPOCHS,
        log_confusion_matrix=CONFUSION_MATRIX,
    )


if __name__ == "__main__":
    main()