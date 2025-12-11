import os
from pathlib import Path
from typing import Dict, Any, Tuple

from ultralytics import YOLO
import cv2
import numpy as np
from tqdm.auto import tqdm

# ---------------- 路径配置 ----------------

FILE = Path(__file__).resolve()
PARTB_DIR = FILE.parent

CONFIG_DIR = PARTB_DIR / "config"
MODEL_DIR = PARTB_DIR / "BasicModel"

DATASET_ROOT = PARTB_DIR.parent / "dataset" / "CrowdHuman"
IMG_ROOT = DATASET_ROOT / "images"
LABEL_ROOT = DATASET_ROOT / "labels"

LOGS_ROOT = PARTB_DIR / "logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)


def get_default_model_path() -> Path:
    """默认 YOLO 预训练权重路径。"""
    return MODEL_DIR / "yolo11s.pt"


def get_default_data_yaml() -> Path:
    """默认 CrowdHuman 数据集配置文件路径。"""
    return CONFIG_DIR / "crowdhuman.yaml"


# ---------------- 模型加载 & 训练 ----------------

def load_model(weights_path: str | Path | None = None) -> YOLO:
    """
    加载 YOLO 模型
    默认加载 BasicModel/yolo11s.pt
    你也可以从外部下载其他 YOLO 权重放进 BasicModel 后传路径进来。
    """
    if weights_path is None:
        weights_path = get_default_model_path()

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    model = YOLO(str(weights_path))
    return model


def train_crowdhuman(
    data_yaml: str | Path | None = None,
    weights: str | Path | None = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    run_name: str | None = None,
):
    """
    在 CrowdHuman YOLO 标注上训练 YOLO11s

    日志目录统一为:
        partB/logs/<run_name>/

    返回:
        model  : 训练后的 YOLO 模型对象
        results: Ultralytics 的训练结果对象
    """
    if data_yaml is None:
        data_yaml = get_default_data_yaml()
    if weights is None:
        weights = get_default_model_path()

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"cannot find dataset configuration: {data_yaml}")

    if run_name is None:
        # 和 FasterRCNN 一样使用时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"yolo11s_crowdhuman_{timestamp}"

    # YOLO 的 project = LOGS_ROOT，name = run_name
    # 这样所有输出都在: LOGS_ROOT/run_name/ 里
    out_dir = LOGS_ROOT / run_name

    print("===> Start training YOLO on CrowdHuman ...")
    print(f" data_yaml: {data_yaml}")
    print(f"   weights: {weights}")
    print(f"   log dir: {out_dir}  (等价于 Ultralytics project='{LOGS_ROOT}', name='{run_name}')")

    model = YOLO(str(weights))

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(LOGS_ROOT),
        name=run_name,
        verbose=True,
    )

    print("===> Training finished.")
    print(f"YOLO 训练日志与权重保存在: {out_dir}")

    return model, results


# ---------------- IoU & 指标计算 ----------------

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算 IoU
    boxes1: [N,4]  x1,y1,x2,y2  (这里是归一化坐标 0~1)
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


def _load_gt_boxes_norm(
    label_path: Path,
    person_class_id: int = 0,
) -> np.ndarray:
    """
    从 YOLO 标签文件读取 GT 框，并转为归一化 xyxy（0~1）
    一行: cls cx cy w h
    """
    boxes = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id != person_class_id:
                continue
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def draw_detections_on_image(
    image_path: str | Path,
    boxes_norm: np.ndarray,
    scores: np.ndarray,
    count: int | None = None,
    score_thresh: float = 0.25,
) -> np.ndarray:
    """
    读入图，使用归一化坐标的预测框画出检测结果和人数统计。
    - boxes_norm: [N,4]  归一化 xyxy
    返回 BGR numpy 图像
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    h, w = img.shape[:2]

    # 绘制 bbox
    for box, score in zip(boxes_norm, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

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


def compute_detection_metrics(
    model: YOLO,
    split: str = "val",
    iou_thresh: float = 0.5,
    score_thresh: float = 0.25,
    person_class_id: int = 0,
    max_vis_images: int = 20,
    vis_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    在指定 split (train/val) 上计算：
      - detection_accuracy = TP / (TP + FN) = TP / total_gt
      - false_positives = FP
      - missed_detections = FN

    不使用自定义 DataLoader，直接用 YOLO 的 predict(stream=True)
    加 tqdm 显示评估进度。
    """
    split = split.lower()
    if split not in ("train", "val"):
        raise ValueError(f"split 必须是 'train' 或 'val'，当前: {split}")

    img_dir = IMG_ROOT / split
    label_dir = LABEL_ROOT / split

    if not img_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if len(img_paths) == 0:
        raise RuntimeError(f"在 {img_dir} 下没有找到任何图像")

    total_gt = 0
    total_pred = 0
    tp = 0
    fp = 0

    vis_count = 0
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 使用 tqdm 包裹图片遍历 —— 和 FasterRCNN 评估时一样有进度条
    for img_path in tqdm(img_paths, desc=f"[YOLO eval] {split}", unit="img"):
        label_path = label_dir / f"{img_path.stem}.txt"

        # GT: 归一化 xyxy
        gt_boxes = _load_gt_boxes_norm(label_path, person_class_id=person_class_id)
        total_gt += len(gt_boxes)

        # 预测
        results = model.predict(
            source=str(img_path),
            conf=score_thresh,
            verbose=False,
        )
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            xyxyn = result.boxes.xyxyn.cpu().numpy()  # 归一化 xyxy

            mask = (cls == person_class_id) & (conf >= score_thresh)
            pred_boxes = xyxyn[mask]
            pred_scores = conf[mask]
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)

        total_pred += len(pred_boxes)

        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)  # [P,G]
            order = np.argsort(-pred_scores)      # 置信度从高到低

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
            people_count = len(pred_boxes)
            vis_img = draw_detections_on_image(
                img_path,
                pred_boxes,
                pred_scores,
                count=people_count,
                score_thresh=score_thresh,
            )
            save_path = vis_dir / img_path.name
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


# ---------------- 单张图片检测（和 FasterRCNN 版类似） ----------------

def detect_people_on_image(
    image_path: str | Path,
    model: YOLO | None = None,
    conf: float = 0.25,
    save_path: str | Path | None = None,
    show: bool = False,
    person_class_id: int = 0,
):
    """
    单张图片检测人：
      - 加载/使用 YOLO 模型
      - 检测所有 person
      - 画框 + 人数统计

    返回:
      people_count: 检测到的人数
      img_bgr: 画好框的 BGR 图像
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"cannot find picture: {image_path}")

    if model is None:
        model = load_model()

    results = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False,
    )

    result = results[0]

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"can not read picture: {image_path}")

    people_count = 0

    if result.boxes is not None and len(result.boxes) > 0:
        cls = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()

        for c, s, box in zip(cls, confs, xyxy):
            if c != person_class_id or s < conf:
                continue

            people_count += 1
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {s:.2f}"
            cv2.putText(
                img_bgr,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        img_bgr,
        f"Count: {people_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img_bgr)

    if show:
        cv2.imshow("YOLO People Detection", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return people_count, img_bgr


# ---------------- 整体评估（风格对齐 FasterRCNN 版） ----------------

def evaluate_best_model_on_val(
    log_dir: str | Path,
    score_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    max_vis_images: int = 50,
    save_subdir: str = "val_best_vis",
) -> Tuple[Dict[str, Any], str, str]:
    """
    使用训练得到的 best.pt 在 CrowdHuman 验证集上做一次完整评估，
    并将可视化结果和指标保存到对应训练目录中。

    参数:
      - log_dir: YOLO 训练日志目录，也就是 train_crowdhuman 对应的:
                 partB/logs/<run_name>/
      - score_thresh: 置信度阈值
      - iou_thresh: IoU 阈值
      - max_vis_images: 最多可视化多少张 val 图像
      - save_subdir: 在 log_dir 下保存可视化的子目录名

    返回:
      - metrics: 字典，包含 detection_accuracy, false_positives, missed_detections 等
      - vis_dir_str: 可视化图像保存的文件夹路径
      - metrics_txt_str: 指标 txt 文件路径
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir 不存在: {log_dir}")

    best_path = log_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"在 {log_dir} 下找不到 weights/best.pt，请确认 YOLO 已训练完成。")

    print(f"[Eval-YOLO] 使用 best.pt: {best_path}")

    model = YOLO(str(best_path))

    vis_dir = log_dir / save_subdir

    metrics = compute_detection_metrics(
        model,
        split="val",
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
        person_class_id=0,
        max_vis_images=max_vis_images,
        vis_dir=vis_dir,
    )

    print(
        "[Eval-YOLO] Detection accuracy: {:.4f}, FP: {}, FN: {}, TP: {}, GT: {}, Pred: {}".format(
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
        f.write("Evaluation on validation set using YOLO best.pt\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"[Eval-YOLO] 指标已保存到: {metrics_txt_path}")
    print(f"[Eval-YOLO] 可视化结果保存在: {vis_dir}")

    return metrics, str(vis_dir), str(metrics_txt_path)
