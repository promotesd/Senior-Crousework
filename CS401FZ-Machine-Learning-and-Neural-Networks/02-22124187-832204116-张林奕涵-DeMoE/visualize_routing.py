import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.options import get_args
from model import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_transform(img_size=(384, 128)):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_model(args, ckpt_path, num_classes=11003):
    model = build_model(args, num_classes)
    model.to(device)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 去掉 module.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def get_visual_routing(model, image_path, transform):
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    outputs = model.base_model.encode_image(img_tensor, l_aux=0, return_routing=True)
    x, l_aux, routing_list = outputs
    return img_pil, routing_list


def infer_patch_grid(num_patches, img_h=384, img_w=128, patch=16, stride=16):
    h_patches = (img_h - patch) // stride + 1
    w_patches = (img_w - patch) // stride + 1
    if h_patches * w_patches == num_patches:
        return h_patches, w_patches
    side = int(math.sqrt(num_patches))
    return side, num_patches // side


def save_original_image(image_pil, save_path):
    image_pil.save(save_path)


def plot_expert_masks(image_pil, routing_list, layer_id=11, num_experts=6, save_path="expert_masks.png"):
    layer_info = routing_list[layer_id]
    selected = layer_info["selected_experts"][:, 0, 0].numpy()  # [L], Top-1
    selected = selected[1:]  # remove CLS token

    h_patches, w_patches = infer_patch_grid(len(selected))
    grid = selected.reshape(h_patches, w_patches)

    orig_w, orig_h = image_pil.size

    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    axes = axes.flatten()

    for e in range(num_experts):
        mask = (grid == e).astype(np.float32)

        ax = axes[e]
        ax.imshow(image_pil)
        ax.imshow(
            mask,
            cmap="Reds",
            alpha=0.45,
            interpolation="nearest",
            extent=[0, orig_w, orig_h, 0]
        )
        ax.set_title(f"Expert {e}", fontsize=11)
        ax.axis("off")

    plt.suptitle(f"Top-1 Expert Masks (Layer {layer_id})", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def collect_usage_matrix(routing_list, num_experts=6):
    """
    返回 [num_layers, num_experts] 的使用次数矩阵
    """
    usage = []
    for layer_info in routing_list:
        sel = layer_info["selected_experts"][:, 0, 0].numpy()  # [L], sample0, top-1
        counts = np.bincount(sel, minlength=num_experts)
        usage.append(counts)
    return np.stack(usage, axis=0)


def plot_expert_usage_heatmap(routing_list, num_experts=6, save_path="expert_usage_heatmap.png"):
    usage = collect_usage_matrix(routing_list, num_experts=num_experts)

    plt.figure(figsize=(7, 5))
    plt.imshow(usage, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="Selection Count")
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.xticks(np.arange(num_experts), [f"E{i}" for i in range(num_experts)])
    plt.yticks(np.arange(usage.shape[0]), np.arange(usage.shape[0]))
    plt.title("Layer-wise Top-1 Expert Usage")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_expert_usage_lines(routing_list, num_experts=6, save_path="expert_usage_lines.png"):
    usage = collect_usage_matrix(routing_list, num_experts=num_experts)

    plt.figure(figsize=(8, 4.5))
    for e in range(num_experts):
        plt.plot(usage[:, e], marker="o", linewidth=2, label=f"Expert {e}")
    plt.xlabel("Layer")
    plt.ylabel("Selection Count")
    plt.title("Expert Usage Across Layers")
    plt.legend(ncol=3, fontsize=9, frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_routing_stats(routing_list, num_experts=6, save_path="routing_stats.txt"):
    usage = collect_usage_matrix(routing_list, num_experts=num_experts)
    mean_usage = usage.mean(axis=0)
    std_usage = usage.std(axis=0)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Routing Statistics ===\n")
        f.write(f"Num layers: {usage.shape[0]}\n")
        f.write(f"Num experts: {num_experts}\n\n")

        f.write("Average Top-1 Usage Per Expert:\n")
        for e in range(num_experts):
            f.write(f"Expert {e}: mean={mean_usage[e]:.3f}, std={std_usage[e]:.3f}\n")

        f.write("\nPer-layer Usage Matrix:\n")
        for l in range(usage.shape[0]):
            f.write(f"Layer {l}: {usage[l].tolist()}\n")


def process_single_image(model, image_path, transform, out_root, num_experts=6, layer_id=11):
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(out_root, img_name)
    os.makedirs(save_dir, exist_ok=True)

    img_pil, routing_list = get_visual_routing(model, image_path, transform)

    save_original_image(img_pil, os.path.join(save_dir, "original.png"))

    plot_expert_masks(
        img_pil,
        routing_list,
        layer_id=layer_id,
        num_experts=num_experts,
        save_path=os.path.join(save_dir, f"expert_masks_layer{layer_id}.png")
    )

    plot_expert_usage_heatmap(
        routing_list,
        num_experts=num_experts,
        save_path=os.path.join(save_dir, "expert_usage_heatmap.png")
    )

    plot_expert_usage_lines(
        routing_list,
        num_experts=num_experts,
        save_path=os.path.join(save_dir, "expert_usage_lines.png")
    )

    save_routing_stats(
        routing_list,
        num_experts=num_experts,
        save_path=os.path.join(save_dir, "routing_stats.txt")
    )

    print(f"[OK] Saved routing visualizations for {image_path} -> {save_dir}")


def main():
    args = get_args()

    args.dataset_name = "RSICD"
    ckpt_path = r"/share/zhangyudong6-nfs/AAAZLYH/code/DeMoE/logs/RSICD/SMA +LB + DR-20251018_214727_baseline/best.pth"

    image_paths = [
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00001.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00002.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00003.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00004.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00005.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00006.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00011.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00008.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00009.jpg",
        r"/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset/RSICD/train/00010.jpg",
        
        
    ]

    out_root = r"/share/zhangyudong6-nfs/AAAZLYH/code/DeMoE/vis_outputs/routing_cases"
    os.makedirs(out_root, exist_ok=True)

    model = load_model(args, ckpt_path)
    transform = build_transform(args.img_size)

    for image_path in image_paths:
        process_single_image(
            model=model,
            image_path=image_path,
            transform=transform,
            out_root=out_root,
            num_experts=args.num_experts,
            layer_id=11,
        )


if __name__ == "__main__":
    main()