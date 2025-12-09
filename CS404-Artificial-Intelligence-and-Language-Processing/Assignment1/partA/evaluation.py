# test.py
import os
import time
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def denormalize_image(
    img: torch.Tensor,
    mean: Optional[List[float]],
    std: Optional[List[float]],
) -> np.ndarray:

    img_np = img.cpu().numpy()  # C x H x W
    if mean is not None and std is not None:
        mean_arr = np.array(mean).reshape(-1, 1, 1)
        std_arr = np.array(std).reshape(-1, 1, 1)
        img_np = img_np * std_arr + mean_arr
    img_np = np.clip(img_np, 0.0, 1.0)

    # C x H x W -> H x W x C
    img_np = np.transpose(img_np, (1, 2, 0))
    return img_np


def visualize_examples(
    examples: List[Tuple[torch.Tensor, int, int]],
    class_names: List[str],
    save_path: str,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    max_samples: int = 16,
    title: str = "",
):

    if len(examples) == 0:
        return

    n = min(len(examples), max_samples)
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))

    for idx in range(n):
        img_tensor, true_label, pred_label = examples[idx]
        img_np = denormalize_image(img_tensor, image_mean, image_std)

        plt.subplot(rows, cols, idx + 1)
        if img_np.shape[2] == 1:
            plt.imshow(img_np[:, :, 0], cmap="gray")
        else:
            plt.imshow(img_np)
        plt.axis("off")
        plt.title(
            f"T:{class_names[true_label]}\nP:{class_names[pred_label]}",
            fontsize=8,
        )

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
):

    cm = conf_mat.astype(np.float32)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    num_classes = len(class_names)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(
    model,
    checkpoint_path: str,
    test_dataset,
    batch_size: int,
    num_classes: int,
    class_names: List[str],
    log_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    max_visualization_samples: int = 16,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
) -> Dict:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log_dir is None:
        log_dir = os.path.dirname(checkpoint_path)

    test_results_dir = os.path.join(log_dir, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)

    model = model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_true = []
    all_pred = []

    correct_examples: List[Tuple[torch.Tensor, int, int]] = []
    incorrect_examples: List[Tuple[torch.Tensor, int, int]] = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

            for i in range(len(labels)):
                img_i = images[i].cpu() 
                t = labels[i].item()
                p = preds[i].item()
                if t == p:
                    if len(correct_examples) < max_visualization_samples:
                        correct_examples.append((img_i, t, p))
                else:
                    if len(incorrect_examples) < max_visualization_samples:
                        incorrect_examples.append((img_i, t, p))

    elapsed = time.time() - start_time

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    overall_acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    cls_report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )
    cls_report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    metrics: Dict = {
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "num_samples": int(len(y_true)),
        "eval_time_sec": float(elapsed),
        "classification_report_dict": cls_report_dict,
    }

    cm_path = os.path.join(test_results_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        conf_mat,
        class_names=class_names,
        save_path=cm_path,
        normalize=True,
        title="Normalized Confusion Matrix",
    )

    correct_path = os.path.join(test_results_dir, "correct_examples.png")
    incorrect_path = os.path.join(test_results_dir, "incorrect_examples.png")

    visualize_examples(
        correct_examples,
        class_names=class_names,
        save_path=correct_path,
        image_mean=image_mean,
        image_std=image_std,
        max_samples=max_visualization_samples,
        title="Correct Predictions",
    )

    visualize_examples(
        incorrect_examples,
        class_names=class_names,
        save_path=incorrect_path,
        image_mean=image_mean,
        image_std=image_std,
        max_samples=max_visualization_samples,
        title="Incorrect Predictions",
    )

    metrics_txt_path = os.path.join(test_results_dir, "test_metrics.txt")
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Num samples: {len(y_true)}\n")
        f.write(f"Evaluation time (s): {elapsed:.2f}\n\n")
        f.write("Classification report:\n")
        f.write(cls_report_str + "\n")

    metrics["log_dir"] = log_dir
    metrics["test_results_dir"] = test_results_dir
    metrics["confusion_matrix_path"] = cm_path
    metrics["correct_examples_path"] = correct_path
    metrics["incorrect_examples_path"] = incorrect_path
    metrics["metrics_txt_path"] = metrics_txt_path

    return metrics
