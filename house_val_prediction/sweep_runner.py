import wandb
from main import main

sweep_config = {
    "method": "grid",  # Możesz zmienić na "random" dla losowego przeszukiwania
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
            "value": [32, 64, 128]
        },
        "NUM_LAYERS": {
            "value": [1, 2, 3]
        },
        "DROPOUT": {
            "value": [0.2, 0.3, 0.4, 0.5]
        },
        "LR": {
            "value": [0.001, 0.0005]
        },
        "WEIGHT_DECAY": {
            "value": [0.0, 0.01]
        },
        "BATCH_SIZE": {
            "value": [32, 64]
        },
        "epochs": {
            "value": 10
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="House Cost")
wandb.agent(sweep_id, function=main)