import torch
import pandas as pd
from pathlib import Path
from config import BEST_CONFIG
from data_preparation import create_data_loaders, create_test_dataloader
from model import HybridNet
from engine import train
import wandb


def main():
    config = BEST_CONFIG
    loss_weights = (config["LOSS_WEIGHT_REG"], 1 - config["LOSS_WEIGHT_REG"])


    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    train_dataloader, val_dataloader, class_weights = create_data_loaders(
        data_path=Path(config["DATA_PATH"]),
        batch_size=config["BATCH_SIZE"],
        val_size=0.2,
    )

    model = HybridNet(
        input_size=config["INPUT_SIZE"],
        hidden_dim=config["HIDDEN_DIM"],
        num_layers=config["NUM_LAYERS"],
        dropout=config["DROPOUT"],
        batch_norm=config["BATCH_NORM"],
    ).to(device)

    loss_fn_reg = torch.nn.MSELoss().to(device)
    loss_fn_class = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5
    )
    wandb.init()
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=(loss_fn_reg, loss_fn_class),
        epochs=config["EPOCHS"],
        device=device,
        model_type="hybrid",
        loss_weights=loss_weights,
    )


    test_dataloader = create_test_dataloader(
        test_data_path=Path(config["TEST_DATA_PATH"]),
        batch_size=config["BATCH_SIZE"],
    )

    model.eval()
    predictions = []
    with torch.inference_mode():
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            _, outputs_class = model(inputs)
            predicted_classes = torch.argmax(outputs_class, dim=1).cpu().numpy()
            predictions.extend(predicted_classes)

    output_path = "pred.csv"
    pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    main()
