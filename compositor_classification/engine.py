import torch
import torch.nn as nn
from tqdm import tqdm

import wandb


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    num_epochs: int,
    class_names: list[str],
    device: torch.device,
) -> None:

    for epoch in tqdm(range(1, num_epochs + 1)):
        loss, accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )
        val_loss, all_preds, all_targets = val_epoch(
            model,
            val_loader,
            loss_fn,
            device,
        )
        scheduler.step()
        val_accuracy = (
            (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
    wandb.log(
        {
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_targets,
                preds=all_preds,
                class_names=class_names,
            )
        }
    )
    wandb.finish()


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device).unsqueeze(-1)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def val_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.to(device).unsqueeze(-1)
            targets = targets.to(device)

            logits = model(inputs)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            loss = loss_fn(logits, targets)

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = total_loss / len(val_loader)

    return avg_loss, all_targets, all_preds
