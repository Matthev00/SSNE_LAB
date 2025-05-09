from pathlib import Path

from utils import set_seeds, weights_init
from data_utils import get_dataloaders, export_real_images_for_fid, build_class_distribution
from models import Generator, Discriminator
from engine import train, compute_fid  
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    """
    Main function to run the script.
    """
    set_seeds()

    # === Hyperparameters ===
    DATA_DIR = Path("trafic_signs/data/trafic_32")
    BATCH_SIZE = 64
    LATENT_DIM = 128
    NUM_CLASSES = 43
    NUM_EPOCHS = 100
    FID_SAMPLE_COUNT = 1000
    MAX_SAMPLES = None
    EMBEDDING_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load data ===
    train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, MAX_SAMPLES, FID_SAMPLE_COUNT)

    # === Models ===
    generator = Generator(nz=LATENT_DIM, ngf=96, num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    discriminator = Discriminator(ndf=96, num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # === Optimizers and schedulers ===
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.5)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    # === FID setup ===
    fid_class_dist = build_class_distribution()
    fid_reference_dir = Path("trafic_signs/data/fid_reference")
    fid_output_dir = Path("trafic_signs/data/fid_outputs")

    print("[INFO] Exporting real validation images for FID...")
    export_real_images_for_fid(val_loader, fid_reference_dir)

    # === Train ===
    print("[INFO] Starting training...")
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
        fid_interval=5
    )


if __name__ == "__main__":
    main()
