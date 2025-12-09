import os
import time
from datetime import datetime
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def train_model(
    model,
    train_dataset,
    test_dataset,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_epochs: int = 20,
    weight_decay: float = 1e-4,
    patience: int = 4,
    min_delta: float = 1e-5,
    log_root: str = "./logs",
    exp_name: str | None = None,
    device: torch.device | None = None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    if exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{model.__class__.__name__}_lr{lr}_bs{batch_size}_{timestamp}"

    exp_name = str(exp_name).replace(" ", "").replace(":", "-")
    log_dir = os.path.join(log_root, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    log_txt_path = os.path.join(log_dir, "train_log.txt")
    best_model_path = os.path.join(log_dir, "model_best.pth")


    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(
            f"lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}, "
            f"weight_decay={weight_decay}, patience={patience}, min_delta={min_delta}\n\n"
        )

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time": [],
    }

    best_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    best_state_dict = None

    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        qbar = tqdm(
            train_dataloader,
            desc=f"[{model.__class__.__name__}] Epoch {epoch+1}/{num_epochs} (lr={lr}, bs={batch_size})",
            leave=False,
        )

        for images, labels in qbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            qbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_dataloader.dataset)

        model.eval()
        test_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_test_batch = criterion(outputs, labels)

                test_loss_sum += loss_test_batch.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_loss = test_loss_sum / total
        test_acc = correct / total

        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)

        log_line = (
            f"[{model.__class__.__name__}] Epoch [{epoch+1}/{num_epochs}] "
            f"LR={lr}, BS={batch_size} | "
            f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
            f"Test Acc={test_acc:.4f}, Time={epoch_time:.2f}s"
        )
        print(log_line)
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        plt.figure()
        plt.plot(history["epoch"], history["test_acc"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.title(f"{model.__class__.__name__} | lr={lr}, bs={batch_size}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "test_acc_epoch.png"))
        plt.close()

        plt.figure()
        plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
        plt.plot(history["epoch"], history["test_loss"], marker="s", label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model.__class__.__name__} | lr={lr}, bs={batch_size}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "loss_epoch.png"))
        plt.close()

        if test_acc > best_acc + min_delta:
            best_acc = test_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, best_model_path)
            best_line = f"New best model at epoch {best_epoch} with acc {best_acc:.4f}"
            print(best_line)
            with open(log_txt_path, "a", encoding="utf-8") as f:
                f.write(best_line + "\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                es_line = (
                    f"[{model.__class__.__name__}] Early stopping at epoch {epoch+1} "
                    f"(best epoch {best_epoch}, best acc {best_acc:.4f})"
                )
                print(es_line)
                with open(log_txt_path, "a", encoding="utf-8") as f:
                    f.write(es_line + "\n")
                break

    total_time = time.time() - total_start
    summary_line = (
        f"[{model.__class__.__name__}] Total training time: {total_time:.2f}s, "
        f"Best Acc={best_acc:.4f} at epoch {best_epoch}"
    )
    print(summary_line)
    with open(log_txt_path, "a", encoding="utf-8") as f:
        f.write(summary_line + "\n")

    history["total_time"] = total_time
    history["best_acc"] = best_acc
    history["best_epoch"] = best_epoch
    history["log_dir"] = log_dir
    history["model_best_path"] = best_model_path if best_state_dict is not None else None
    history["log_txt_path"] = log_txt_path

    return history, log_dir
