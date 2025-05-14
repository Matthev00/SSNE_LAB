import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import wandb
from tqdm import tqdm

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    num_epochs: int,
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
        val_loss, val_accuracy = val_epoch(
            model,
            val_loader,
            loss_fn,
            device,
        )
        scheduler.step()

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
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

    hidden, state = model.init_hidden(next(iter(train_loader))[0].size(0))
    hidden, state = hidden.to(device), state.to(device)
    for inputs, targets, input_len in tqdm(train_loader):
        inputs = inputs.to(device).unsqueeze(-1)
        targets = targets.to(device)
        
        # hidden, state = model.init_hidden(inputs.size(0))
        # hidden, state = hidden.to(device), state.to(device)

        optimizer.zero_grad()

        inputs_packed = pack_padded_sequence(inputs, input_len, batch_first=True, enforce_sorted=False)

        preds, (hidden, state) = model(inputs_packed, (hidden, state))
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted_classes = torch.max(preds, 1)
        total_correct += (predicted_classes == targets).sum().item()
        total_samples += targets.size(0)

        hidden = hidden.detach()
        state = state.detach()

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def val_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    hidden, state = model.init_hidden(next(iter(val_loader))[0].size(0))
    hidden, state = hidden.to(device), state.to(device)
    with torch.no_grad():
        for inputs, targets, input_len in tqdm(val_loader):
            inputs = inputs.to(device).unsqueeze(-1)
            targets = targets.to(device)
            
            # hidden, state = model.init_hidden(inputs.size(0))
            # hidden, state = hidden.to(device), state.to(device)

            inputs_packed = pack_padded_sequence(inputs, input_len, batch_first=True, enforce_sorted=False)
            preds, (hidden, state) = model(inputs_packed, (hidden, state))
            loss = loss_fn(preds, targets)

            total_loss += loss.item()
            _, predicted_classes = torch.max(preds, 1)
            total_correct += (predicted_classes == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy