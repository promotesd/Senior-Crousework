# Part B — CrowdHuman People Detection (Faster R-CNN & YOLOv11)

## Table of Contents

1. Project Structure
2. Dataset Preparation

   1. Download CrowdHuman
   2. Why CrowdHuman (instead of COCO / PASCAL VOC)
   3. Dataset Layout
   4. Convert `.odgt` to YOLO labels
3. File Overview
4. Evaluation Metrics (What results contain)
5. Why Faster R-CNN & YOLOv11

---

## 1. Project Structure

The project uses the following directory layout:

```bash
Assignment1/
├── dataset/CrowdHuman/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── annotation_train.odgt
│   └── annotation_val.odgt
└── partB/
```

> The dataset is stored under `Assignment1/dataset/CrowdHuman/`, and all training / evaluation code and outputs are under `Assignment1/partB/`.

---

## 2. Dataset Preparation

### (1) CrowdHuman dataset link

CrowdHuman Dataset download link:
[https://aistudio.baidu.com/datasetdetail/89331](https://aistudio.baidu.com/datasetdetail/89331)

---

### (2) Why only CrowdHuman (instead of COCO / PASCAL VOC)

This PartB focuses on **people detection and crowd scenarios**. CrowdHuman is chosen because:

* **Crowd-focused annotations**: CrowdHuman contains dense crowds, heavy occlusion, and large scale variation—exactly the challenging cases for person detection.
* **More suitable than COCO for “crowd person counting/detection”**: COCO includes “person” class but is more general-purpose; dense crowd scenes are less dominant.
* **More suitable than PASCAL VOC**: VOC is relatively small and old, and does not cover modern crowd density / occlusion patterns well.

Therefore, to keep the task focused and realistic for crowded person detection, this project uses **CrowdHuman**.

---

### (3) Dataset location

The dataset is placed in:

```bash
Assignment1/
├── dataset/CrowdHuman/
│   ├── images/train, images/val
│   ├── labels/train, labels/val
│   ├── annotation_train.odgt, annotation_val.odgt
└── partB/
```

---

### (4) Convert `.odgt` annotations to YOLO labels

This project uses **YOLO-format labels** (`class cx cy w h`, normalized to `[0,1]`) under:

* `dataset/CrowdHuman/labels/train/`
* `dataset/CrowdHuman/labels/val/`

Conversion commands:

```bash
cd \Assignment1\partB
python odgt2yolo.py ../dataset/CrowdHuman/annotation_train.odgt
python odgt2yolo.py --val ../dataset/CrowdHuman/annotation_val.odgt
```

After conversion, label files will be generated as `labels/train/*.txt` and `labels/val/*.txt`, aligned by image filename.

---

## 3. File Overview

### Core scripts / notebooks

* **`FasterRCNN_crowdhuman.py`**
  Training + evaluation utilities for Faster R-CNN (ResNet50-FPN v2) on CrowdHuman.

* **`FasterRCNN_train_eval.ipynb`**
  Notebook for running Faster R-CNN training and evaluation experiments.

* **`yolo_crowdhuman.py`**
  Training + evaluation utilities for YOLOv11 on CrowdHuman (Ultralytics).

* **`yolo11_train_eval.ipynb`**
  Notebook for running YOLOv11 training and evaluation experiments.

### Other folders

* **`config/`**
  Stores the YOLO dataset config file `crowdhuman.yaml`.

* **`logs/`**
  Stores experiment outputs (training logs, best weights, validation visualizations, metrics text files).

* **`single_test/`**
  Stores single-image test results (visualized prediction images).

> In addition to images, logs typically include metric summaries (e.g., `val_best_metrics.txt`) and per-run training logs.

---

## 4. Evaluation Metrics (What results contain)

For both Faster R-CNN and YOLO, evaluation includes the following metrics:

* **`true_positives (TP)`**
  Number of predicted boxes that correctly match a ground-truth person box (IoU ≥ `iou_thresh`).

* **`false_positives (FP)`**
  Number of predicted boxes that do **not** match any ground-truth person box (or duplicate matches).
  (Typically caused by over-detection, background confusion, or multiple boxes on the same person.)

* **`missed_detections (FN)`**
  Number of ground-truth persons that were not detected (TP did not cover them).

* **`total_gt`**
  Total number of ground-truth person boxes in the evaluation split.

* **`total_pred`**
  Total number of predicted person boxes after filtering by confidence threshold.

* **`detection_accuracy`**
  Defined as recall on ground-truth persons:

  
  detection_accuracy = TP/(TP + FN)
  

### Visualization output (val_best)

During evaluation, each validation image visualization includes:

* Predicted bounding boxes + confidence score
* **Predicted count (Pred)** and **Ground-truth count (GT)** shown at top-left

---

## 5. Why Faster R-CNN and YOLOv11

This project uses **two complementary detectors** to compare different detection paradigms:

### Faster R-CNN (Two-stage detector)

* Strong performance on **high-quality localization**
* Often more stable in hard cases (occlusion / crowded scenes)
* Useful baseline for academic-style detection pipelines

### YOLOv11 (One-stage detector)

* Very fast inference speed
* Strong practical performance and easy deployment
* Modern YOLO variants perform well on crowd detection with good recall–speed tradeoff

---
## 6. How to run 
After dataset preparation.
### Faster R-CNN
Run code blocks in FasterRCNN_train_eval.ipynb one by one
### YOLOv11
Run code blocks in yolo11_train_eval.ipynb one by one