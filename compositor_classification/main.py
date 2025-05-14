from data_utils import get_data_loaders
from model import LSTMClassifier
from engine import train

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import wandb

def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_SIZE = 1
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 5
    LR = 0.001
    BATCH_SIZE = 50
    NUM_EPOCHS = 10
    STEP_SIZE = 5
    GAMMA = 0.1

    model = LSTMClassifier(
        input_size=INPUT_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=OUTPUT_SIZE
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LR
    )
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=STEP_SIZE, 
        gamma=GAMMA
    )

    wandb.init(
        project="compositor_classification",
        config={
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "output_size": OUTPUT_SIZE,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "step_size": STEP_SIZE,
            "gamma": GAMMA
        }
    )

    train_dataloader, val_dataloader = get_data_loaders(BATCH_SIZE)

    train(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        device=device
    )



if __name__ == "__main__":
    main()