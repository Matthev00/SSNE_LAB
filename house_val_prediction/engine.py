import torch
import wandb
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    log_confusion_matrix: bool = False,
    device: str = "cuda",
) -> dict[str, list[float]]:

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy, train_f1 = train_step(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_loss, val_accuracy, val_f1, all_targets, all_predictions  = test_step(
            model=model, val_dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "Train Accuracy": train_accuracy,
                "Validation Accuracy": val_accuracy,
                "Train F1": train_f1,
                "Validation F1": val_f1,
            }
        )

    if log_confusion_matrix:
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
        wandb.log({"Confusion Matrix": wandb.Image("confusion_matrix.png")})

    wandb.finish()


def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device="cuda",
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    for batch, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_targets, all_predictions, average="weighted")
    total_loss /= len(train_dataloader)

    return total_loss, accuracy, f1


def test_step(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device="cuda",
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.inference_mode():
        for batch, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_targets, all_predictions, average="weighted")
    total_loss /= len(val_dataloader)

    return total_loss, accuracy, f1, all_targets, all_predictions
