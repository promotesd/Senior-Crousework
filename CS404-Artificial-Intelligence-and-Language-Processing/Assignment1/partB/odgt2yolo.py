# odgt2yolo.py
# convert CrowdHuman .odgt annotations to YOLO format

# 目录结构为：
# Assignment1/
# ├── dataset/CrowdHuman/
# │   ├── images/train, images/val
# │   ├── labels/train, labels/val
# │   ├── annotation_train.odgt, annotation_val.odgt
# └── partB/odgt2yolo.py

import json
from pathlib import Path
from PIL import Image
import argparse

FILE = Path(__file__).resolve()

PROJECT_ROOT = FILE.parents[1]

DATA_ROOT = PROJECT_ROOT / "dataset" / "CrowdHuman"

IMG_ROOT = DATA_ROOT / "images"
LABEL_ROOT = DATA_ROOT / "labels"


def convert_to_yolo_format(opt):
    odgt_path = Path(opt.path)

    if not odgt_path.exists():
        raise FileNotFoundError(f"odgt 文件不存在: {odgt_path}")

    print(f"使用 odgt: {odgt_path}")
    print(f"数据集根目录: {DATA_ROOT}")

    which = "val" if opt.val else "train"

    img_dir = IMG_ROOT / which
    label_dir = LABEL_ROOT / which
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取图片目录: {img_dir}")
    print(f"输出标签目录: {label_dir}")

    with odgt_path.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue

            aitem = json.loads(line)
            img_id = aitem["ID"]

            image_path = img_dir / f"{img_id}.jpg"
            if not image_path.exists():
                print(f"[警告] 找不到图片: {image_path}，跳过这一条")
                continue

            label_path = label_dir / f"{img_id}.txt"

            img = Image.open(image_path)
            img_w, img_h = img.width, img.height

            gtboxes = aitem.get("gtboxes", [])

            with label_path.open("w", encoding="utf-8") as wf:
                for af in gtboxes:

                    if af.get("tag") == "mask":
                        continue

                    extra = af.get("extra", {})

                    if extra.get("ignore", 0) == 1:
                        continue


                    vbox = af.get("vbox", None)
                    if vbox is not None:
                        x, y, w, h = vbox 
                        cx = x + w / 2.0
                        cy = y + h / 2.0

                        wf.write(
                            f"0 {cx / img_w:.6f} {cy / img_h:.6f} "
                            f"{w / img_w:.6f} {h / img_h:.6f}\n"
                        )


                    # head_attr = af.get("head_attr", {})

                    # if head_attr.get("ignore", 0) == 0 and "hbox" in af:
                    #     hbox = af["hbox"]
                    #     x, y, w, h = hbox
                    #     cx = x + w / 2.0
                    #     cy = y + h / 2.0

                    #     wf.write(
                    #         f"1 {cx / img_w:.6f} {cy / img_h:.6f} "
                    #         f"{w / img_w:.6f} {h / img_h:.6f}\n"
                    #     )

            img.close()


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val",
        action="store_true",
        help="处理验证集 (annotation_val.odgt, images/val, labels/val)",
    )


    parser.add_argument("path", help="odgt 标注文件路径")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    convert_to_yolo_format(opt)
