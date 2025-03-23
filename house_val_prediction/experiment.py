import torch
import wandb
from data_preparation import create_data_loaders
from model import HouseNet, HybridNet


def run_experiment(config=None):
    with wandb.init(config=config):
        config = wandb.config

        device = "cuda" if torch.cuda.is_available() else "cpu"

        train_loader, val_loader, class_weights = create_data_loaders(
            data_path="train_data.csv",
            batch_size=config.batch_size,
            val_size=0.2,
        )

        if config.model_type == "classifier":
            model = HouseNet(
                input_size=25,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                output_size=3,
                dropout=config.dropout,
            ).to(device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

        elif config.model_type == "regressor":
            model = HouseNet(
                input_size=25,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                output_size=1,
                dropout=config.dropout,
            ).to(device)
            loss_fn = torch.nn.MSELoss().to(device)

        elif config.model_type == "hybrid":
            model = HybridNet(
                input_size=25,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            ).to(device)
            loss_fn_reg = torch.nn.MSELoss().to(device)
            loss_fn_class = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Trening
        if config.model_type == "hybrid":
            train_hybrid(model, train_loader, val_loader, optimizer, loss_fn_reg, loss_fn_class, config.epochs, device)
        else:
            train(model, train_loader, val_loader, optimizer, loss_fn, config.epochs, device)