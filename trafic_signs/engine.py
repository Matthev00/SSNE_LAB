from pathlib import Path
import subprocess
from typing import Callable

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import denormalize


def train_epoch(
    generator,
    discriminator,
    train_loader,
    generator_optimizer,
    discriminator_optimizer,
    criterion,
    latent_dim,
    num_classes,
    device,
) -> tuple[float, float]:
    generator.train()
    discriminator.train()
    cls_criterion = nn.CrossEntropyLoss()

    g_loss_total = 0.0
    d_loss_total = 0.0
    d_loss_real_total = 0.0
    d_loss_fake_total = 0.0

    for real_images, real_labels in train_loader:
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        b_size = real_images.size(0)

        noise = torch.randn(b_size, latent_dim, device=device)
        ones = torch.ones(b_size, 1, device=device)
        zeros = torch.zeros(b_size, 1, device=device)

        # === Generator ===
        generator_optimizer.zero_grad()

        fake_images = generator(noise, real_labels)
        d_fake_adv, d_fake_cls = discriminator(fake_images, real_labels)

        g_adv_loss = criterion(d_fake_adv, ones)
        g_cls_loss = cls_criterion(d_fake_cls, real_labels)
        g_loss = g_adv_loss + g_cls_loss

        g_loss.backward()
        generator_optimizer.step()

        # === Discriminator ===
        discriminator_optimizer.zero_grad()

        d_real_adv, d_real_cls = discriminator(real_images, real_labels)
        d_loss_real = criterion(d_real_adv, ones)

        d_fake_adv, _ = discriminator(fake_images.detach(), real_labels)
        d_loss_fake = criterion(d_fake_adv, zeros)

        d_cls_loss = cls_criterion(d_real_cls, real_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2 + d_cls_loss
        d_loss.backward()
        discriminator_optimizer.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        d_loss_real_total += d_loss_real.item()
        d_loss_fake_total += d_loss_fake.item()

    return (
        g_loss_total / len(train_loader),
        d_loss_total / len(train_loader),
        d_loss_real_total / len(train_loader),
        d_loss_fake_total / len(train_loader),
    )


def test_epoch(
    generator: nn.Module,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
    fid_sample_count: int,
    fid_class_dist: dict[int, float],
    fid_output_dir: Path,
    epoch: int,
    fid_reference_dir: Path,
    compute_fid_fn: Callable[[str, str], float],
) -> float:
    import numpy as np
    from torchvision.transforms import ToPILImage

    generator.eval()
    out_dir = fid_output_dir / f"epoch_{epoch}" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.random.choice(
        list(fid_class_dist.keys()),
        size=fid_sample_count,
        p=[fid_class_dist[k] for k in fid_class_dist],
    )

    with torch.inference_mode():
        for i, label in enumerate(labels):
            z = torch.randn(64, latent_dim, device=device)
            l = torch.full((64,), label, device=device, dtype=torch.long)
            imgs = generator(z, l).cpu()

            for j, img in enumerate(imgs):
                img = denormalize(img)
                img = img.clamp(0, 1).mul(255).to(torch.uint8)
                img = img.cpu()
                to_pil = ToPILImage()
                to_pil(img).save(out_dir / f"{i:04}_{j:02}.png")

    return compute_fid_fn(str(fid_reference_dir), str(out_dir))


def train(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scheduler: Callable,
    discriminator_scheduler: Callable,
    criterion: Callable,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
    num_epochs: int,
    fid_class_dist: dict[int, float],
    fid_sample_count: int,
    fid_output_dir: Path,
    fid_reference_dir: Path,
    compute_fid_fn: Callable[[str, str], float],
    fid_interval: int = 5,
) -> tuple[list[float], list[float], list[float]]:

    G_losses, D_losses, FID_scores = [], [], []

    for epoch in range(1, num_epochs + 1):
        g_loss, d_loss, d_loss_real, d_loss_fake = train_epoch(
            generator,
            discriminator,
            train_loader,
            generator_optimizer,
            discriminator_optimizer,
            criterion,
            latent_dim,
            num_classes,
            device,
        )

        generator_scheduler.step()
        discriminator_scheduler.step()

        G_losses.append(g_loss)
        D_losses.append(d_loss)
        wandb.log(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
                "d_loss_real": d_loss_real,
                "d_loss_fake": d_loss_fake,
            },
            step=epoch,
        )

        print(f"[Epoch {epoch:03}] G_loss: {g_loss:.4f} | D_loss: {d_loss:.4f}")

        if epoch % fid_interval == 0:
            fid = test_epoch(
                generator,
                latent_dim,
                num_classes,
                device,
                fid_sample_count,
                fid_class_dist,
                fid_output_dir,
                epoch,
                fid_reference_dir,
                compute_fid_fn,
            )
            FID_scores.append(fid)
            print(f"[FID] Epoch {epoch}: FID = {fid:.2f}")
            wandb.log(
                {
                    "FID": fid,
                },
                step=epoch,
            )

    wandb.finish()
    return G_losses, D_losses, FID_scores


def compute_fid(real_dir: str, generated_dir: str) -> float:
    """
    Computes the FID score between two directories of images.

    Args:
        real_dir (str): Path to the directory containing real images.
        generated_dir (str): Path to the directory containing generated images.

    Returns:
        float: The FID score.

    Raises:
        RuntimeError: If the FID calculation fails or the result is not found.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", real_dir, generated_dir],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "FID:" in line:
                return float(line.strip().split()[-1])
    except subprocess.CalledProcessError as e:
        print("[ERROR] FID subprocess failed:\n", e.stderr)
        raise RuntimeError("Failed to compute FID")

    raise RuntimeError("FID result not found in output")
