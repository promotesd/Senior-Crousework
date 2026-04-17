import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from textwrap import fill
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.options import get_args
from model import build_model
from datasets.RSICD import RSICD
from datasets.RSITMD import RSITMD
from datasets.Sydney_captions import Sydney_captions
from datasets.UCM_captions import UCM_captions
from datasets.bases import tokenize
from utils.simple_tokenizer import SimpleTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_FACTORY = {
    "RSICD": RSICD,
    "RSITMD": RSITMD,
    "Sydney_captions": Sydney_captions,
    "UCM_captions": UCM_captions,
}


def build_transform(img_size=(384, 128)):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def robust_load_model(args, ckpt_path, num_classes=11003):
    model = build_model(args, num_classes)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 常见 checkpoint 格式兼容
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 去掉可能的 "module." 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint from: {ckpt_path}")
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def load_dataset(args):
    if args.dataset_name not in DATASET_FACTORY:
        raise ValueError(
            f"Unsupported dataset_name: {args.dataset_name}. "
            f"Available: {list(DATASET_FACTORY.keys())}"
        )
    dataset_cls = DATASET_FACTORY[args.dataset_name]
    dataset = dataset_cls(root=args.root_dir, verbose=False)
    ds = dataset.test if args.val_dataset == "test" else dataset.val
    return ds


@torch.no_grad()
def encode_all_images(model, img_paths: List[str], transform, batch_size: int = 128):
    all_feats = []
    imgs = []

    for p in tqdm(img_paths, desc="Loading images"):
        img = Image.open(p).convert("RGB")
        img = transform(img)
        imgs.append(img)

    for i in tqdm(range(0, len(imgs), batch_size), desc="Encoding images"):
        batch = torch.stack(imgs[i:i + batch_size], dim=0).to(device)
        feat = model.encode_image(batch, l_aux=0)
        if isinstance(feat, tuple):
            feat = feat[0]
        feat = feat.float()

        if feat.dim() == 3:
            feat = feat[:, 0, :]

        feat = F.normalize(feat, dim=-1)
        all_feats.append(feat.cpu())

    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def encode_all_texts(model, captions: List[str], text_length: int = 77, batch_size: int = 256):
    tokenizer = SimpleTokenizer()
    all_tokens = []

    for cap in tqdm(captions, desc="Tokenizing texts"):
        tok = tokenize(cap, tokenizer, text_length=text_length)
        all_tokens.append(tok)

    all_feats = []
    for i in tqdm(range(0, len(all_tokens), batch_size), desc="Encoding texts"):
        batch_tokens = torch.stack(all_tokens[i:i + batch_size], dim=0).to(device)
        feat = model.encode_text(batch_tokens, l_aux=0)
        if isinstance(feat, tuple):
            feat = feat[0]
        feat = feat.float()

        # IRRA 返回 [B, D]；底层 CLIP 可能返回 [B, L, D]
        if feat.dim() == 3:
            eot_pos = batch_tokens.argmax(dim=-1)
            feat = feat[torch.arange(feat.shape[0]), eot_pos]

        feat = F.normalize(feat, dim=-1)
        all_feats.append(feat.cpu())

    return torch.cat(all_feats, dim=0)


def draw_t2i_case(case_id: int,
                  query_caption: str,
                  gt_img_path: str,
                  topk_img_paths: List[str],
                  correct_flags: List[bool],
                  topk_scores: List[float],
                  save_path: str):
    k = len(topk_img_paths)
    fig = plt.figure(figsize=(3 * (k + 1), 4.2))

    # GT Image
    ax0 = fig.add_subplot(1, k + 1, 1)
    ax0.imshow(Image.open(gt_img_path).convert("RGB"))
    ax0.axis("off")
    ax0.set_title("GT Image", fontsize=10)

    for i, (img_path, ok, score) in enumerate(zip(topk_img_paths, correct_flags, topk_scores), start=2):
        ax = fig.add_subplot(1, k + 1, i)
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.axis("off")
        color = "green" if ok else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax.set_title(f"Top-{i-1}\n{score:.3f}", fontsize=10, color=color)

    fig.suptitle(f"Case {case_id}\nQuery Text:\n" + fill(query_caption, width=90), fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def collect_t2i_cases(ds,
                      sim_t2i: torch.Tensor,
                      num_correct: int = 5,
                      num_wrong: int = 5,
                      topk: int = 5):
    correct_cases = []
    wrong_cases = []

    used_correct_gt = set()
    used_correct_top1 = set()
    used_correct_signature = set()

    used_wrong_gt = set()
    used_wrong_top1 = set()
    used_wrong_signature = set()

    num_queries = sim_t2i.shape[0]

    for q_idx in range(num_queries):
        scores, topk_idx = torch.topk(sim_t2i[q_idx], k=topk)
        topk_idx = topk_idx.tolist()
        topk_scores = scores.tolist()

        query_caption = ds["captions"][q_idx]
        gt_img_path = ds["caption_img_paths"][q_idx]
        topk_img_paths = [ds["img_paths"][i] for i in topk_idx]
        correct_flags = [p == gt_img_path for p in topk_img_paths]
        top1_img_path = topk_img_paths[0]
        retrieval_signature = tuple(topk_img_paths[:3])

        case = {
            "q_idx": q_idx,
            "query_caption": query_caption,
            "gt_img_path": gt_img_path,
            "topk_img_paths": topk_img_paths,
            "correct_flags": correct_flags,
            "topk_scores": topk_scores,
        }

        if correct_flags[0]:
            if (
                gt_img_path not in used_correct_gt
                and top1_img_path not in used_correct_top1
                and retrieval_signature not in used_correct_signature
                and len(correct_cases) < num_correct
            ):
                correct_cases.append(case)
                used_correct_gt.add(gt_img_path)
                used_correct_top1.add(top1_img_path)
                used_correct_signature.add(retrieval_signature)
        else:
            if (
                gt_img_path not in used_wrong_gt
                and top1_img_path not in used_wrong_top1
                and retrieval_signature not in used_wrong_signature
                and len(wrong_cases) < num_wrong
            ):
                wrong_cases.append(case)
                used_wrong_gt.add(gt_img_path)
                used_wrong_top1.add(top1_img_path)
                used_wrong_signature.add(retrieval_signature)

        if len(correct_cases) >= num_correct and len(wrong_cases) >= num_wrong:
            break

    return correct_cases, wrong_cases


def main():
    args = get_args()

    args.dataset_name = "RSICD"   # RSITMD / Sydney_captions / UCM_captions 
    args.val_dataset = "test"

    ckpt_path = r"/share/zhangyudong6-nfs/AAAZLYH/code/DeMoE/logs/RSICD/SMA +LB + DR-20251018_214727_baseline/best.pth"
    topk = 5
    num_correct = 5
    num_wrong = 5

    ds = load_dataset(args)
    model = robust_load_model(args, ckpt_path)
    transform = build_transform(args.img_size)

    print(f"[INFO] Dataset: {args.dataset_name}")
    print(f"[INFO] #Images: {len(ds['img_paths'])} | #Captions: {len(ds['captions'])}")

    img_feats = encode_all_images(model, ds["img_paths"], transform, batch_size=128)   # [Ng, D]
    txt_feats = encode_all_texts(model, ds["captions"], args.text_length, batch_size=256)  # [Nq, D]

    sim_t2i = txt_feats @ img_feats.t()  # [Nq, Ng]

    correct_cases, wrong_cases = collect_t2i_cases(
        ds, sim_t2i,
        num_correct=num_correct,
        num_wrong=num_wrong,
        topk=topk
    )

    out_dir = os.path.join("vis_outputs", "retrieval_cases")
    os.makedirs(out_dir, exist_ok=True)

    # 正确案例
    for i, case in enumerate(correct_cases, start=1):
        save_path = os.path.join(out_dir, f"correct_{i}.png")
        draw_t2i_case(
            case_id=i,
            query_caption=case["query_caption"],
            gt_img_path=case["gt_img_path"],
            topk_img_paths=case["topk_img_paths"],
            correct_flags=case["correct_flags"],
            topk_scores=case["topk_scores"],
            save_path=save_path
        )

    # 错误案例
    for i, case in enumerate(wrong_cases, start=1):
        save_path = os.path.join(out_dir, f"wrong_{i}.png")
        draw_t2i_case(
            case_id=i,
            query_caption=case["query_caption"],
            gt_img_path=case["gt_img_path"],
            topk_img_paths=case["topk_img_paths"],
            correct_flags=case["correct_flags"],
            topk_scores=case["topk_scores"],
            save_path=save_path
        )

    print(f"[INFO] Saved {len(correct_cases)} correct cases and {len(wrong_cases)} wrong cases to: {out_dir}")


if __name__ == "__main__":
    main()