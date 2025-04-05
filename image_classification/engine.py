import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import wandb


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_fn: torch.nn.Module,
    epochs: int,
    class_names: list[str],
    log_confusion_matrix: bool = False,
    device: str = "cuda",
) -> dict[str, list[float]]:

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss, val_accuracy, all_preds, all_targets = test_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "Train Accuracy": train_accuracy,
                "Validation Accuracy": val_accuracy,
            }
        )

        scheduler.step(val_loss)

        if log_confusion_matrix:
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix (Epoch {epoch+1})")
            plt.tight_layout()
            path = f"confusion_matrix_epoch_{epoch+1}.png"
            plt.savefig(path)
            plt.close()
            wandb.log({f"Confusion Matrix (Epoch {epoch+1})": wandb.Image(path)})

    wandb.finish()


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device="cuda",
) -> tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred_logit = model(X)

        loss = loss_fn(y_pred_logit, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(input=y_pred_logit, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / y.size(0)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device="cuda",
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    test_loss, test_acc = 0, 0
    all_preds, all_targets = [], []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred_logit = model(X)

            loss = loss_fn(y_pred_logit, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(input=y_pred_logit, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / y.size(0)

            all_preds.extend(y_pred_class.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc, all_preds, all_targets
