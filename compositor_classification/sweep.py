from main import main

import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "avg_accuracy", "goal": "maximize"},
    "parameters": {
        "input_size": {"value": 1},
        "output_size": {"value": 5},
        "hidden_size": {"values": [32, 64, 128, 256, 512]},
        "num_layers": {"values": [2, 3, 4, 5]},
        "lr": {"values": [0.001, 0.0001, 0.00001]},
        "batch_size": {"values": [16, 32, 64]},
        "num_epochs": {"value": 10},
        "step_size": {"value": 5},
        "gamma": {"value": 0.1},
        "dropout": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "weight_decay": {"values": [0.0, 1e-5, 1e-4, 1e-3]},
        "scheduler_type": {"values": ["StepLR"]},
        "bidirectional": {"values": [True, False]},
        "embedding_dim": {"values": [32, 64, 128]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="compositor_classification")
    wandb.agent(sweep_id, function=main, count=100)
