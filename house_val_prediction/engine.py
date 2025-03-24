import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from tqdm import tqdm
from pickle import load

import wandb


def encode_value(x):
    scaler = load(open("y_scaler.pkl", "rb"))
    x = scaler.inverse_transform(x)
    if x <= 100000:
        return 0
    elif 100000 < x <= 350000:
        return 1
    else:
        return 2


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_fn: torch.nn.Module,
    epochs: int,
    log_confusion_matrix: bool = False,
    device: str = "cuda",
    model_type="classifier",
    loss_weights: tuple[float, float] = (1.0, 1.0),
) -> dict[str, list[float]]:

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy, train_f1, train_balanced_accuracy = train_step(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            model_type=model_type,
            loss_weights=loss_weights
        )
        val_loss, val_accuracy, val_f1, all_targets, all_predictions, val_balanced_accuracy = test_step(
            model=model,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            model_type=model_type,
            loss_weights=loss_weights
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
                "Train Balanced Accuracy": train_balanced_accuracy,
                "Validation Balanced Accuracy": val_balanced_accuracy,
            }
        )
        scheduler.step(val_loss)

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
    model_type="classifier",
    loss_weights: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    for inputs, targets_reg, targets_class in train_dataloader:
        inputs = inputs.to(device)

        if model_type == "classifier":
            targets = targets_class.to(device)
        elif model_type == "regressor":
            targets = targets_reg.to(device).float()

        optimizer.zero_grad()

        if model_type == "hybrid":
            outputs_reg, outputs_class = model(inputs)
            regression_loss = loss_fn[0](outputs_reg.squeeze(), targets_reg.to(device))
            classification_loss = loss_fn[1](outputs_class, targets_class.to(device))
            loss = regression_loss * loss_weights[0] + classification_loss * loss_weights[1]
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if model_type == "classifier" or model_type == "hybrid":
            predicted = torch.argmax(
                torch.softmax(outputs_class if model_type == "hybrid" else outputs, 1),
                1,
            )
            correct += (predicted == targets_class.to(device)).sum().item()
            total += targets_class.size(0)
            all_targets.extend(targets_class.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        elif model_type == "regressor":
            predicted_classes = [
                encode_value(x) for x in outputs.cpu().detach().numpy()
            ]
            true_classes = [encode_value(x) for x in targets_reg.cpu().numpy()]

            all_targets.extend(true_classes)
            all_predictions.extend(predicted_classes)

            correct += sum(p == t for p, t in zip(predicted_classes, true_classes))
            total += len(true_classes)

    accuracy = correct / total if total > 0 else 0
    f1 = f1_score(all_targets, all_predictions, average="weighted") if total > 0 else 0
    balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions) if total > 0 else 0
    total_loss /= len(train_dataloader)

    return total_loss, accuracy, f1, balanced_accuracy


def test_step(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device="cuda",
    model_type="classifier",
    loss_weights: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.inference_mode():
        for inputs, targets_reg, targets_class in val_dataloader:
            inputs = inputs.to(device)

            if model_type == "classifier":
                targets = targets_class.to(device)
            elif model_type == "regressor":
                targets = targets_reg.to(device)

            if model_type == "hybrid":
                outputs_reg, outputs_class = model(inputs)
                regression_loss = loss_fn[0](
                    outputs_reg.squeeze(), targets_reg.to(device)
                )
                classification_loss = loss_fn[1](
                    outputs_class, targets_class.to(device)
                )
                loss = regression_loss * loss_weights[0] + classification_loss * loss_weights[1]
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs.squeeze(), targets)

            total_loss += loss.item()

            if model_type == "classifier" or model_type == "hybrid":
                predicted = torch.argmax(
                    torch.softmax(
                        outputs_class if model_type == "hybrid" else outputs, 1
                    ),
                    1,
                )
                correct += (predicted == targets_class.to(device)).sum().item()
                total += targets_class.size(0)
                all_targets.extend(targets_class.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            elif model_type == "regressor":
                predicted_classes = [encode_value(x) for x in outputs.cpu().numpy()]
                true_classes = [encode_value(x) for x in targets_reg.cpu().numpy()]

                all_targets.extend(true_classes)
                all_predictions.extend(predicted_classes)

                correct += sum(p == t for p, t in zip(predicted_classes, true_classes))
                total += len(true_classes)

    accuracy = correct / total if total > 0 else 0
    f1 = f1_score(all_targets, all_predictions, average="weighted") if total > 0 else 0
    balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions) if total > 0 else 0
    total_loss /= len(val_dataloader)

    return total_loss, accuracy, f1, all_targets, all_predictions, balanced_accuracy
