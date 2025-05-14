from pathlib import Path

from utils import set_seeds, weights_init
from data_utils import (
    get_dataloaders,
    export_real_images_for_fid,
    build_class_distribution,
)
from models import Generator, Discriminator
from engine import train, compute_fid
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


def main():
    """
    Main function to run the script.
    """
    set_seeds()

    # === Hyperparameters ===
    NUM_CLASSES = 43
    FID_SAMPLE_COUNT = 1000
    FID_INTERVAL = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path("trafic_signs/data/trafic_32/trafic_32")
    fid_reference_dir = Path("trafic_signs/data/fid_reference")
    fid_output_dir = Path("trafic_signs/data/fid_outputs")
    fid_class_dist = build_class_distribution()

    # Lista eksperymentów
    # experiments = [
    #     {"name": "latent_64", "LATENT_DIM": 64},
    #     {"name": "embedding_128", "EMBEDDING_DIM": 128},
    #     {"name": "batch_32", "BATCH_SIZE": 32},
    #     {"name": "epochs_200", "NUM_EPOCHS": 200},
    #     {"name": "ngf_ndf_256", "NGF": 256, "NDF": 256},
    #     {"name": "scheduler_fast_decay", "SCHEDULER_STEP": 10, "SCHEDULER_GAMMA": 0.3},
    #     {"name": "loss_mse", "CRITERION": "mse"},
    #     {"name": "latent_256", "LATENT_DIM": 256},
    #     {"name": "embedding_32", "EMBEDDING_DIM": 32},
    #     {"name": "batch_128", "BATCH_SIZE": 128},
    # ]

    # Uruchom eksperymenty
    # for exp in experiments:
        # Ustaw domyślne hiperparametry
    LATENT_DIM = 128
    EMBEDDING_DIM = 64
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    NGF = 128
    NDF = 128
    SCHEDULER_STEP = 30
    SCHEDULER_GAMMA = 0.5

    # Dane
    train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, None, FID_SAMPLE_COUNT)
    export_real_images_for_fid(val_loader, fid_reference_dir)

    # Model
    generator = Generator(LATENT_DIM, NGF, NUM_CLASSES, EMBEDDING_DIM).to(DEVICE)
    discriminator = Discriminator(NDF, NUM_CLASSES, EMBEDDING_DIM).to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optymalizacja
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # W&B
    wandb.init(
        project="trafic_signs",
        config={
            "latent_dim": LATENT_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "ngf": NGF,
            "ndf": NDF,
            "scheduler_step": SCHEDULER_STEP,
            "scheduler_gamma": SCHEDULER_GAMMA,
        },
    )

    # Trening
    G_losses, D_losses, FID_scores = train(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        generator_optimizer=g_optimizer,
        discriminator_optimizer=d_optimizer,
        generator_scheduler=g_scheduler,
        discriminator_scheduler=d_scheduler,
        criterion=criterion,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        fid_class_dist=fid_class_dist,
        fid_sample_count=FID_SAMPLE_COUNT,
        fid_output_dir=fid_output_dir,
        fid_reference_dir=fid_reference_dir,
        compute_fid_fn=compute_fid,
        fid_interval=FID_INTERVAL,
    )



if __name__ == "__main__":
    main()
