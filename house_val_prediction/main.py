import torch
import wandb

from data_preparation import create_data_loaders
from engine import train
from model import HouseNet


def main():
    torch.manual_seed(42)
    EPOCHS = 10
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    CLS = True
    DROPOUT = 0.5
    BATCH_SIZE = 32
    LR = 0.001
    WEIGHT_DECAY = 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, val_dataloader, class_weights = create_data_loaders(
        data_path="train_data.csv",
        batch_size=BATCH_SIZE,
        classification=CLS,
        val_size=0.2,
    )

    model = HouseNet(
        input_size=25,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_size=3 if CLS else 1,
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
    )


if __name__ == "__main__":
    main()
