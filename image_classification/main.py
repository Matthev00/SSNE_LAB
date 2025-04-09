import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from engine import train
from models import CNN_BN, LeNetPlus, TinyResNet, TinyVGG

import wandb
from data import get_data_loaders

MODEL_MAP = {
    "LeNetPlus": LeNetPlus,
    "TinyVGG": TinyVGG,
    "CNN_BN": CNN_BN,
    "TinyResNet": TinyResNet,
}


def set_seed(seed: int = 42):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run():
    set_seed()
    wandb.init()
    config = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DATA
    data_dir = Path("data/train")
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        data_percent=config.data_percent,
        val_split=0.2,
    )
    class_names = train_loader.dataset.dataset.classes

    # MODEL
    model_class = MODEL_MAP[config.model_name]
    model = model_class(
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        num_classes=len(class_names),
    )
    model = model.to(device)

    # OPTIMIZER
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # LR SCHEDULER
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.patience, factor=config.scheduler_factor
    )

    # LOSS
    loss_fn = nn.CrossEntropyLoss()

    # TRAIN
    train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=config.epochs,
        class_names=class_names,
        log_confusion_matrix=True,
        device=device,
    )
    # Log model
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(model.state_dict(), tmp.name)

        artifact = wandb.Artifact(
            name=f"{config.hidden_size}_model",
            type="model",
            description="Trained model weights",
            metadata=dict(config),
        )
        artifact.add_file(tmp.name)
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    run()
