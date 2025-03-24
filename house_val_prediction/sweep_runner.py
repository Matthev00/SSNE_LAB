import wandb
from main import main

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "Validation F1",
        "goal": "maximize",
    },
    "parameters": {
        "MODEL_TYPE": {
            "values": ["classifier", "regressor", "hybrid"]
        },
        "BATCH_NORM": {
            "values": [True, False]
        },
        "HIDDEN_DIM": {
            "value": [32, 64, 128, 256, 512]
        },
        "NUM_LAYERS": {
            "value": [1, 2, 3, 4]
        },
        "DROPOUT": {
            "value": [0.2, 0.3, 0.4, 0.5, 0.7]
        },
        "LR": {
            "value": [1e-3, 5e-4, 1e-4]
        },
        "WEIGHT_DECAY": {
            "value": [0.0, 1e-2, 1e-3]
        },
        "BATCH_SIZE": {
            "value": [32, 64]
        },
        "epochs": {
            "value": [10, 20]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="House Cost")
wandb.agent(sweep_id, function=main)