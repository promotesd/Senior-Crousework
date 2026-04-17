import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def load_dataset(args):
    dataset_cls = DATASET_FACTORY[args.dataset_name]
    dataset = dataset_cls(root=args.root_dir, verbose=False)
    ds = dataset.test if args.val_dataset == "test" else dataset.val
    return ds


def decode_tokens(token_tensor, tokenizer):
    inv_vocab = {v: k for k, v in tokenizer.encoder.items()}
    ids = token_tensor.tolist()
    tokens = []
    for tid in ids:
        if tid == 0:
            continue
        tok = inv_vocab.get(tid, f"<id:{tid}>")
        tokens.append(tok)
    return tokens


def clean_token_string(tok: str):
    tok = tok.replace("</w>", "")
    tok = tok.replace("<|startoftext|>", "<SOT>")
    tok = tok.replace("<|endoftext|>", "<EOT>")
    tok = tok.replace("<|mask|>", "<MASK>")
    return tok


@torch.no_grad()
def get_text_routing(model, caption: str, text_length=77):
    tokenizer = SimpleTokenizer()
    token_tensor = tokenize(caption, tokenizer, text_length=text_length).unsqueeze(0).to(device)

    outputs = model.base_model.encode_text(token_tensor, l_aux=0, return_routing=True)
    x, l_aux, routing_list = outputs

    token_list = decode_tokens(token_tensor[0].cpu(), tokenizer)
    token_list = [clean_token_string(t) for t in token_list]

    return token_tensor[0].cpu(), token_list, routing_list


@torch.no_grad()
def get_image_routing(model, image_path, transform):
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    outputs = model.base_model.encode_image(img_tensor, l_aux=0, return_routing=True)
    x, l_aux, routing_list = outputs
    return img_pil, routing_list


def compute_layer_entropy(routing_list):
    """
    每层平均路由熵
    gate_logits: [L, B, E]
    返回: [num_layers]
    """
    entropies = []
    for layer_info in routing_list:
        gate_logits = layer_info["gate_logits"].float()   # [L, B, E]
        probs = torch.softmax(gate_logits, dim=-1)        # [L, B, E]
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [L, B]
        entropies.append(entropy.mean().item())
    return np.array(entropies)


def compute_token_layer_top1_matrix(routing_list, remove_cls=False):
    """
    返回 [num_layers, num_tokens]
    对应每层每个 token 的 top-1 expert id
    """
    mats = []
    for layer_info in routing_list:
        selected = layer_info["selected_experts"][:, 0, 0].cpu().numpy()  # [L]
        mats.append(selected)
    mats = np.stack(mats, axis=0)  # [num_layers, num_tokens]

    if remove_cls:
        mats = mats[:, 1:]

    return mats


def collect_usage_matrix(routing_list, num_experts):
    """
    每层专家使用次数 [num_layers, num_experts]
    """
    usage = []
    for layer_info in routing_list:
        selected = layer_info["selected_experts"][:, 0, 0].cpu().numpy()
        counts = np.bincount(selected, minlength=num_experts)
        usage.append(counts)
    return np.stack(usage, axis=0)




def plot_text_token_layer_routing(token_list, routing_list, save_path, remove_sot=False):
    mat = compute_token_layer_top1_matrix(routing_list, remove_cls=False)  # [layer, token]

    if remove_sot and len(token_list) == mat.shape[1]:
        token_list = token_list[1:]
        mat = mat[:, 1:]

    plt.figure(figsize=(max(10, len(token_list) * 0.35), 5))
    plt.imshow(mat, aspect="auto", cmap="tab10")
    plt.colorbar(label="Top-1 Expert ID")
    plt.yticks(np.arange(mat.shape[0]), [f"L{i}" for i in range(mat.shape[0])])
    plt.xticks(np.arange(len(token_list)), token_list, rotation=60, ha="right", fontsize=8)
    plt.xlabel("Token")
    plt.ylabel("Layer")
    plt.title("Text Routing: Token × Layer Top-1 Expert")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_text_expert_usage_heatmap(routing_list, num_experts, save_path):
    usage = collect_usage_matrix(routing_list, num_experts=num_experts)

    plt.figure(figsize=(7, 5))
    plt.imshow(usage, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="Selection Count")
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.xticks(np.arange(num_experts), [f"E{i}" for i in range(num_experts)])
    plt.yticks(np.arange(usage.shape[0]), [f"L{i}" for i in range(usage.shape[0])])
    plt.title("Text Routing: Layer-wise Expert Usage")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_single_entropy_curve(entropy_values, title, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(entropy_values)), entropy_values, marker="o", linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("Average Routing Entropy")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_compare_entropy_curve(img_entropy, txt_entropy, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(img_entropy)), img_entropy, marker="o", linewidth=2, label="Image Branch")
    plt.plot(np.arange(len(txt_entropy)), txt_entropy, marker="s", linewidth=2, label="Text Branch")
    plt.xlabel("Layer")
    plt.ylabel("Average Routing Entropy")
    plt.title("Routing Entropy Across Layers")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_dataset_average_entropy(all_img_entropy, all_txt_entropy, save_path):
    """
    输入:
      all_img_entropy: [N, num_layers]
      all_txt_entropy: [N, num_layers]
    """
    img_mean = np.mean(all_img_entropy, axis=0)
    img_std = np.std(all_img_entropy, axis=0)

    txt_mean = np.mean(all_txt_entropy, axis=0)
    txt_std = np.std(all_txt_entropy, axis=0)

    x = np.arange(len(img_mean))

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, img_mean, marker="o", linewidth=2, label="Image Mean Entropy")
    plt.fill_between(x, img_mean - img_std, img_mean + img_std, alpha=0.2)

    plt.plot(x, txt_mean, marker="s", linewidth=2, label="Text Mean Entropy")
    plt.fill_between(x, txt_mean - txt_std, txt_mean + txt_std, alpha=0.2)

    plt.xlabel("Layer")
    plt.ylabel("Average Routing Entropy")
    plt.title("Dataset-level Routing Entropy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()



def save_token_list(token_list, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for i, tok in enumerate(token_list):
            f.write(f"{i}\t{tok}\n")


def save_entropy_stats(img_entropy, txt_entropy, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Routing Entropy Statistics ===\n\n")
        f.write("Image Branch:\n")
        for i, v in enumerate(img_entropy):
            f.write(f"Layer {i}: {v:.6f}\n")
        f.write("\nText Branch:\n")
        for i, v in enumerate(txt_entropy):
            f.write(f"Layer {i}: {v:.6f}\n")


def process_single_sample(model,
                          image_path,
                          caption,
                          transform,
                          out_root,
                          sample_name,
                          num_experts,
                          text_length=77):
    save_dir = os.path.join(out_root, sample_name)
    os.makedirs(save_dir, exist_ok=True)

    # image routing
    img_pil, img_routing = get_image_routing(model, image_path, transform)
    img_pil.save(os.path.join(save_dir, "original_image.png"))

    # text routing
    token_tensor, token_list, txt_routing = get_text_routing(model, caption, text_length=text_length)

    # save raw text
    with open(os.path.join(save_dir, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    save_token_list(token_list, os.path.join(save_dir, "text_tokens.txt"))

    # routing heatmap for text
    plot_text_token_layer_routing(
        token_list,
        txt_routing,
        save_path=os.path.join(save_dir, "text_routing_token_layer.png"),
        remove_sot=False
    )

    plot_text_expert_usage_heatmap(
        txt_routing,
        num_experts=num_experts,
        save_path=os.path.join(save_dir, "text_expert_usage_heatmap.png")
    )

    # entropy
    img_entropy = compute_layer_entropy(img_routing)
    txt_entropy = compute_layer_entropy(txt_routing)

    plot_single_entropy_curve(
        img_entropy,
        title="Image Routing Entropy",
        save_path=os.path.join(save_dir, "image_entropy_per_layer.png")
    )

    plot_single_entropy_curve(
        txt_entropy,
        title="Text Routing Entropy",
        save_path=os.path.join(save_dir, "text_entropy_per_layer.png")
    )

    plot_compare_entropy_curve(
        img_entropy,
        txt_entropy,
        save_path=os.path.join(save_dir, "routing_entropy_compare.png")
    )

    save_entropy_stats(
        img_entropy,
        txt_entropy,
        save_path=os.path.join(save_dir, "routing_entropy_stats.txt")
    )

    return img_entropy, txt_entropy



def main():
    args = get_args()

    args.dataset_name = "RSICD"
    args.val_dataset = "test"

    ckpt_path = r"/share/zhangyudong6-nfs/AAAZLYH/code/DeMoE/logs/RSICD/SMA +LB + DR-20251018_214727_baseline/best.pth"
    out_root = r"/share/zhangyudong6-nfs/AAAZLYH/code/DeMoE/vis_outputs/routing_analysis"
    os.makedirs(out_root, exist_ok=True)

    selected_indices = [0, 1, 2, 3, 4]


    ds = load_dataset(args)
    model = load_model(args, ckpt_path)
    transform = build_transform(args.img_size)

    all_img_entropy = []
    all_txt_entropy = []

    for idx in tqdm(selected_indices, desc="Processing samples"):
        image_path = ds["caption_img_paths"][idx]
        caption = ds["captions"][idx]
        sample_name = f"sample_{idx:04d}"

        img_entropy, txt_entropy = process_single_sample(
            model=model,
            image_path=image_path,
            caption=caption,
            transform=transform,
            out_root=out_root,
            sample_name=sample_name,
            num_experts=args.num_experts,
            text_length=args.text_length,
        )

        all_img_entropy.append(img_entropy)
        all_txt_entropy.append(txt_entropy)

    all_img_entropy = np.stack(all_img_entropy, axis=0)
    all_txt_entropy = np.stack(all_txt_entropy, axis=0)

    plot_dataset_average_entropy(
        all_img_entropy,
        all_txt_entropy,
        save_path=os.path.join(out_root, "dataset_average_entropy.png")
    )

    print(f"Saved routing analysis to: {out_root}")


if __name__ == "__main__":
    main()