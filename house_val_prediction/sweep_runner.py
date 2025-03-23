import wandb
from main import main

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "Validation F1",
        "goal": "maximize",
    },
    "parameters": {
        "HIDDEN_DIM": {
            "values": [32, 64, 128]
        },
        "NUM_LAYERS": {
            "values": [1, 2, 3]
        },
        "DROPOUT": {
            "values": [0.2, 0.3, 0.4, 0.5]
        },
        "LR": {
            "values": [0.001, 0.0005]
        },
        "WEIGHT_DECAY": {
            "values": [0.0, 0.01]
        },
        "BATCH_SIZE": {
            "values": [32, 64]
        },
        "epochs": {
            "value": 10
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="House Cost")
wandb.agent(sweep_id, function=main)
