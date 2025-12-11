import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]          # Assignment1/
DATA_ROOT = PROJECT_ROOT / "dataset" / "CrowdHuman"
IMG_ROOT = DATA_ROOT / "images"
LABEL_ROOT = DATA_ROOT / "labels"

PARTB_DIR = FILE.parent                 # Assignment1/partB
LOGS_ROOT = PARTB_DIR / "logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

torch.multiprocessing.set_sharing_strategy("file_system")
np.random.seed(42)

def create_fasterrcnn_resnet50_fpn_v2(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    创建 Faster R-CNN ResNet50 FPN V2 模型，并替换分类头。

    num_classes: 包含背景。比如 [background, person] => num_classes = 2
    """
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class CrowdHumanYOLODataset(Dataset):
    """
    从 YOLO 标签读取 CrowdHuman 数据：
    - images_dir: .../images/train 或 .../images/val
    - labels_dir: .../labels/train 或 .../labels/val
    YOLO 一行: class cx cy w h (归一化到 [0,1])
    我们只用 class=0，映射到 Faster R-CNN 的 label=1 (person)
    """

    def __init__(self, images_dir: Path, labels_dir: Path, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        # 也许有 .png，根据需要加上：
        self.image_paths += sorted(self.images_dir.glob("*.png"))

        assert len(self.image_paths) > 0, f"未在 {self.images_dir} 找到图像"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        # Label file
        label_path = self.labels_dir / (img_path.stem + ".txt")

        # 读取图像
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = parts
                    cls_id = int(cls_id)
                    cx = float(cx) * w
                    cy = float(cy) * h
                    bw = float(bw) * w
                    bh = float(bh) * h

                    x1 = cx - bw / 2.0
                    y1 = cy - bh / 2.0
                    x2 = cx + bw / 2.0
                    y2 = cy + bh / 2.0

                    # 只保留 person 类，映射到 label=1
                    if cls_id == 0:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)  # person = 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        else:
            img = F.to_tensor(img)

        return img, target, str(img_path)


def collate_fn(batch):
    images, targets, paths = list(zip(*batch))
    return list(images), list(targets), list(paths)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    boxes1: [N,4] x1,y1,x2,y2
    boxes2: [M,4]
    return: [N,M] IoU
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / np.clip(union, a_min=1e-6, a_max=None)
    return iou


def compute_detection_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
    max_vis_images: int = 20,
    vis_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    在验证集上计算：
      - detection_accuracy = TP / (TP + FN) = TP / total_gt
      - false_positives = FP
      - missed_detections = FN

    """
    model.eval()
    total_gt = 0
    total_pred = 0
    tp = 0
    fp = 0

    vis_count = 0
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, targets, paths in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img_tensor, target, output, path in zip(images, targets, outputs, paths):
                gt_boxes = target["boxes"].cpu().numpy()  # [G,4]
                total_gt += len(gt_boxes)

                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()

                mask = (labels == 1) & (scores >= score_thresh)
                pred_boxes = pred_boxes[mask]
                scores = scores[mask]

                total_pred += len(pred_boxes)

                matched_gt = np.zeros(len(gt_boxes), dtype=bool)
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)  # [P,G]
                    order = np.argsort(-scores)
                    for p_idx in order:
                        iou_row = ious[p_idx]
                        g_idx = np.argmax(iou_row)
                        if iou_row[g_idx] >= iou_thresh and not matched_gt[g_idx]:
                            tp += 1
                            matched_gt[g_idx] = True
                        else:
                            fp += 1
                else:
                    fp += len(pred_boxes)

                # 可视化
                if vis_dir is not None and vis_count < max_vis_images:
                    vis_img = draw_detections_on_image(
                        path, pred_boxes, scores, person_label=1, count=None
                    )
                    save_path = vis_dir / Path(path).name
                    cv2.imwrite(str(save_path), vis_img)
                    vis_count += 1

    fn = total_gt - tp
    if total_gt > 0:
        detection_accuracy = tp / total_gt
    else:
        detection_accuracy = 0.0

    metrics = {
        "detection_accuracy": detection_accuracy,
        "false_positives": fp,
        "missed_detections": fn,
        "true_positives": tp,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }
    return metrics



def draw_detections_on_image(
    image_path: str,
    pred_boxes: np.ndarray,
    scores: np.ndarray,
    person_label: int = 1,
    count: int | None = None,
    score_thresh: float = 0.5,
) -> np.ndarray:
    """
    读入图，画出预测框和人数统计。
    返回 BGR numpy 图像
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 绘制 bbox
    for box, score in zip(pred_boxes, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"person {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if count is not None:
        cv2.putText(
            img,
            f"People: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def detect_people_in_image(
    model: torch.nn.Module,
    device: torch.device,
    image_path: str,
    score_thresh: float = 0.5,
    save_path: str | None = None,
) -> Tuple[int, np.ndarray]:
    """
    单张图片推理：
      - 输入: 模型、设备、图片路径
      - 输出: (人数, 画好框的图像 BGR)
    """
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    boxes = output["boxes"].cpu().numpy()

    mask = (labels == 1) & (scores >= score_thresh)
    boxes = boxes[mask]
    scores = scores[mask]
    people_count = len(boxes)

    vis_img = draw_detections_on_image(
        image_path, boxes, scores, person_label=1, count=people_count, score_thresh=score_thresh
    )

    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis_img)

    return people_count, vis_img


def train_fasterrcnn_crowdhuman(
    num_epochs: int = 5,
    batch_size: int = 4,
    lr: float = 0.005,
    num_workers: int = 4,
    device: str | torch.device | None = None,
    run_name: str | None = None,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    weights_path: str | None = None,
) -> Tuple[torch.nn.Module, str, Dict[str, Any]]:
    """
    训练 Faster R-CNN 在 CrowdHuman 上，只输出 Detection accuracy 和 FP/FN。

    日志目录: partB/logs/<run_name>/
      - train_results.txt: 每个 epoch 的 train_loss, detection_accuracy, FP, FN 等
      - val_vis/epoch_xxx/*.jpg: 每个 epoch 部分验证图片的可视化结果
      - best_model.pth: detection_accuracy 最好的模型
    """
    # 设备
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Run 名称
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"fasterrcnn_resnet50_fpn_v2_{timestamp}"

    out_dir = LOGS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(out_dir)

    print(f"日志目录: {out_dir_str}")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"Train images: {IMG_ROOT / 'train'}")
    print(f"Val images  : {IMG_ROOT / 'val'}")

    # 数据集 & DataLoader
    train_dataset = CrowdHumanYOLODataset(
        images_dir=IMG_ROOT / "train",
        labels_dir=LABEL_ROOT / "train",
    )
    val_dataset = CrowdHumanYOLODataset(
        images_dir=IMG_ROOT / "val",
        labels_dir=LABEL_ROOT / "val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}\n")

    # 模型
    NUM_CLASSES = 2  # [background, person]
    if weights_path is None:
        print("加载 COCO 预训练权重")
        model = create_fasterrcnn_resnet50_fpn_v2(num_classes=NUM_CLASSES, pretrained=True)

    else:
        print(f"从本地权重加载模型: {weights_path}")
        model = create_fasterrcnn_resnet50_fpn_v2(num_classes=NUM_CLASSES, pretrained=False)

        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    # 记录
    train_loss_history = []
    acc_history = []
    fp_history = []
    fn_history = []

    best_acc = 0.0
    best_model_path = out_dir / "best_model.pth"

    # train_results.txt
    results_txt_path = out_dir / "train_results.txt"
    with results_txt_path.open("w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,detection_accuracy,false_positives,missed_detections,"
            "true_positives,total_gt,total_pred\n"
        )

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        epoch_bar = tqdm(
            total=len(train_dataset),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            unit="img"
        )

        for step, (images, targets, paths) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            epoch_loss += loss_value
            num_batches += 1

            # 当前 batch 的样本数
            bs = len(images)
            # 更新进度条：前进 bs 张图片
            epoch_bar.update(bs)

            if (step + 1) % 20 == 0:
                tqdm.write(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"Step {step+1}/{len(train_loader)} "
                    f"Loss: {loss_value:.4f}"
                )

        epoch_bar.close()

        epoch_loss /= max(1, num_batches)
        print(f"[Epoch {epoch+1}] 平均训练损失: {epoch_loss:.4f}")

        # 验证集评估
        # vis_dir = out_dir / "val_vis" / f"epoch_{epoch+1:03d}"
        metrics = compute_detection_metrics(
            model,
            val_loader,
            device=device,
            iou_thresh=iou_thresh,
            score_thresh=score_thresh,
            max_vis_images=0,
            vis_dir=None,
        )

        acc = metrics["detection_accuracy"]
        fp = metrics["false_positives"]
        fn = metrics["missed_detections"]

        train_loss_history.append(epoch_loss)
        acc_history.append(acc)
        fp_history.append(fp)
        fn_history.append(fn)

        print(
            f"[Epoch {epoch+1}] "
            f"Detection accuracy: {acc:.4f}, "
            f"FP: {fp}, FN: {fn}, TP: {metrics['true_positives']}, "
            f"GT: {metrics['total_gt']}, Pred: {metrics['total_pred']}"
        )

        # 写入 train_results.txt
        with results_txt_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},{epoch_loss:.6f},{acc:.6f},{fp},{fn},"
                f"{metrics['true_positives']},{metrics['total_gt']},{metrics['total_pred']}\n"
            )

        # 按 detection_accuracy 保存 best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"更新 best model, detection_accuracy={best_acc:.4f}, 保存到 {best_model_path}")

    history = {
        "train_loss": train_loss_history,
        "detection_accuracy": acc_history,
        "false_positives": fp_history,
        "missed_detections": fn_history,
        "best_model": str(best_model_path),
        "log_dir": out_dir_str,
        "results_txt": str(results_txt_path),
    }

    return model, out_dir_str, history




def evaluate_best_model_on_val(
    log_dir: str | Path,
    device: str | torch.device | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    save_subdir: str = "val_best_vis",
) -> Tuple[Dict[str, Any], str, str]:
    '''
    使用训练得到的 best_model.pth 在 CrowdHuman 验证集上做一次完整评估，
    并将所有可视化结果和指标保存到对应训练目录中。

    参数:
      - log_dir: 训练时的日志目录 (train_fasterrcnn_crowdhuman 返回的 log_dir)
      - device: 设备, 如 "cuda:0" 或 torch.device(...)，默认自动选择
      - batch_size, num_workers: 验证用的 DataLoader 配置
      - score_thresh: 置信度阈值, 低于此分数的框会被丢弃
      - iou_thresh: 匹配 IoU 阈值, 默认为 0.5
      - save_subdir: 在 log_dir 下保存可视化的子目录名

    返回:
      - metrics: 一个字典，包含 detection_accuracy, false_positives, missed_detections 等
      - vis_dir_str: 可视化图像保存的文件夹路径 (str)
      - metrics_txt_str: 指标 txt 文件路径 (str)
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    log_dir = Path(log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir 不存在: {log_dir}")

    best_model_path = log_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"在 {log_dir} 下找不到 best_model.pth, 请确认训练已完成并保存了最优模型。")

    print(f"[Eval] 使用 best model: {best_model_path}")

    NUM_CLASSES = 2  # [background, person]
    model = create_fasterrcnn_resnet50_fpn_v2(num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    val_dataset = CrowdHumanYOLODataset(
        images_dir=IMG_ROOT / "val",
        labels_dir=LABEL_ROOT / "val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print(f"[Eval] 验证样本数: {len(val_dataset)}")


    vis_dir = log_dir / save_subdir

    metrics = compute_detection_metrics(
        model,
        val_loader,
        device=device,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
        max_vis_images=len(val_dataset),
        vis_dir=vis_dir,
    )

    print(
        "[Eval] Detection accuracy: {:.4f}, FP: {}, FN: {}, TP: {}, GT: {}, Pred: {}".format(
            metrics["detection_accuracy"],
            metrics["false_positives"],
            metrics["missed_detections"],
            metrics["true_positives"],
            metrics["total_gt"],
            metrics["total_pred"],
        )
    )

    metrics_txt_path = log_dir / "val_best_metrics.txt"
    with metrics_txt_path.open("w", encoding="utf-8") as f:
        f.write("Evaluation on validation set using best_model.pth\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"[Eval] 指标已保存到: {metrics_txt_path}")
    print(f"[Eval] 可视化结果保存在: {vis_dir}")

    return metrics, str(vis_dir), str(metrics_txt_path)
