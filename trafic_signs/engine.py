from pathlib import Path
import subprocess
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import denormalize


def train_epoch(
    generator, discriminator, train_loader,
    generator_optimizer, discriminator_optimizer,
    criterion, latent_dim, num_classes, device
) -> tuple[float, float]:
    
    class_loss_fn = nn.CrossEntropyLoss()

    generator.train()
    discriminator.train()

    g_loss_total = 0.0
    d_loss_total = 0.0

    for real_images, real_labels in train_loader:
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        b_size = real_images.size(0)

        # === Discriminator ===
        discriminator_optimizer.zero_grad()

        real_target = torch.full((b_size,), 0.9, device=device)
        fake_target = torch.zeros(b_size, device=device)

        real_output, real_class_logits = discriminator(real_images, real_labels)
        d_real_loss = criterion(real_output, real_target)
        d_class_loss = class_loss_fn(real_class_logits, real_labels)

        noise = torch.randn(b_size, latent_dim, device=device)
        fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
        with torch.no_grad():
            fake_images = generator(noise, fake_labels)
        fake_output, _ = discriminator(fake_images, fake_labels)
        d_fake_loss = criterion(fake_output, fake_target)

        d_loss = d_real_loss + d_fake_loss + d_class_loss
        d_loss.backward()
        discriminator_optimizer.step()

        # === Generator ===
        generator_optimizer.zero_grad()

        noise = torch.randn(b_size, latent_dim, device=device)
        fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
        fake_images = generator(noise, fake_labels)
        fake_output, fake_class_logits = discriminator(fake_images, fake_labels)

        g_adv_loss = criterion(fake_output, torch.ones(b_size, device=device))
        g_class_loss = class_loss_fn(fake_class_logits, fake_labels)

        g_loss = g_adv_loss + g_class_loss
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        generator_optimizer.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    return g_loss_total / len(train_loader), d_loss_total / len(train_loader)

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
            z = torch.randn(1, latent_dim, device=device)
            l = torch.tensor([label], device=device)
            img = generator(z, l).squeeze(0).cpu()

            img = (denormalize(img).clamp(0, 1) * 255).byte()

            ToPILImage()(img).save(out_dir / f"{i:04}.png")

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
    """
    Główna funkcja trenująca GAN przez wiele epok, z obliczaniem FID co N epok.

    Returns:
        G_losses (List[float])
        D_losses (List[float])
        FID_scores (List[float])
    """
    G_losses, D_losses, FID_scores = [], [], []

    for epoch in range(1, num_epochs + 1):
        g_loss, d_loss = train_epoch(
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
            check=True
        )
        for line in result.stdout.splitlines():
            if "FID:" in line:
                return float(line.strip().split()[-1])
    except subprocess.CalledProcessError as e:
        print("[ERROR] FID subprocess failed:\n", e.stderr)
        raise RuntimeError("Failed to compute FID")

    raise RuntimeError("FID result not found in output")
