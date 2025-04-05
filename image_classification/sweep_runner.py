import wandb
from main import run

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "Validation Accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "model_name": {"values": ["LeNetPlus", "TinyVGG", "CNN_BN", "TinyResNet"]},
        "hidden_size": {"values": [8, 32, 64]},
        "dropout": {"values": [0.5]},
        "learning_rate": {"values": [1e-4]},
        "weight_decay": {"values": [1e-3]},
        "batch_size": {"values": [16, 32]},
        "epochs": {"values": [2]},
        "data_percent": {"values": [0.2]},
        "patience": {"values": [2]},
        "scheduler_factor": {"values": [0.5]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="image-classification", entity="MY_EXPERIMENTS")
wandb.agent(sweep_id, function=run)