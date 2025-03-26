import wandb
from main import main

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "Validation Balanced Accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "MODEL_TYPE": {"values": ["hybrid"]},
        "BATCH_NORM": {"values": [True, False]},
        "HIDDEN_DIM": {"values": [32, 64, 128, 256, 512]},
        "NUM_LAYERS": {"values": [1, 2, 3, 4]},
        "DROPOUT": {"values": [0.2, 0.3, 0.4, 0.5, 0.7]},
        "LR": {"values": [1e-3, 5e-4, 1e-4]},
        "WEIGHT_DECAY": {"values": [0.0, 1e-2, 1e-3]},
        "BATCH_SIZE": {"values": [32, 64]},
        "epochs": {"values": [10, 20]},
        "LOSS_WEIGHT_REG": {"values": [num / 10 for num in range(0, 11)]},
    },
}


sweep_id = wandb.sweep(sweep_config, project="House Cost")
wandb.agent(sweep_id, function=main)
