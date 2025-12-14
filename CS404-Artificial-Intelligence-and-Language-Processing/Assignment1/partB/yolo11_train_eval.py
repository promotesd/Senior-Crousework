from pathlib import Path
from yolo_crowdhuman import (
    train_crowdhuman,
    evaluate_best_model_on_val,
    LOGS_ROOT,  # partB/logs 根目录
)


if __name__ == "__main__":
    EPOCHS = 32
    IMG_SIZE = 640
    BATCH_SIZE = 8

    SCORE_THRESH = 0.25
    IOU_THRESH = 0.5
    MAX_VIS_IMAGES = 50

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    before_runs = {d.name for d in LOGS_ROOT.iterdir() if d.is_dir()}

    print(f"Epochs   : {EPOCHS}")
    print(f"Image sz : {IMG_SIZE}")
    print(f"Batch    : {BATCH_SIZE}")
    print()

    model, results = train_crowdhuman(
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        run_name=None,
    )

    # log_dir = "xxxx"



    # metrics, vis_dir, metrics_txt = evaluate_best_model_on_val(
    #     log_dir=log_dir,
    #     score_thresh=SCORE_THRESH,
    #     iou_thresh=IOU_THRESH,
    #     max_vis_images=MAX_VIS_IMAGES,
    #     save_subdir="val_best_vis", 
    # )

    # print()
    # print("========== [YOLO11 Eval] 评估完成 ==========")
    # print(f"Detection accuracy: {metrics['detection_accuracy']:.4f}")
    # print(f"False positives   : {metrics['false_positives']}")
    # print(f"Missed detections : {metrics['missed_detections']}")
    # print(f"True positives    : {metrics['true_positives']}")
    # print(f"Total GT boxes    : {metrics['total_gt']}")
    # print(f"Total Pred boxes  : {metrics['total_pred']}")
    # print()
    # print(f"可视化结果目录: {vis_dir}")
    # print(f"指标 txt 文件 : {metrics_txt}")


