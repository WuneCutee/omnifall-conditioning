"""
Tạo bảng + hình cuối cùng cho bài báo:
  1. Confusion matrix 4 lớp (fall/lie_down/fallen/lying) — Baseline vs Hybrid v8, PNG 300dpi.
  2. Bar chart 3 subplot (Fallen-F1, Fall-F1, Fall∪Fallen) so sánh Baseline/Rule-based/Hybrid v8
     qua 3 seed, có error bar.

Usage:
    python -m scripts.make_final_figures
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner  # noqa: E402
from src.utils import compute_all_metrics  # noqa: E402

BASELINE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "baseline_logits" / "OOPS_test_stsyn.pkl"
RULE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs" / "OOPS_test_stsyn.pkl"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
               "stand_up", "standing", "other"]
NUM_CLASSES = 10
SEEDS = [42, 43, 44]

# Số liệu điểm/mean-std đã biết (baseline, rule-based deterministic; hybrid v8 = 3 seed thật)
BASELINE = {"fallen": 26.6, "fall": 45.3, "fall_union": 55.5}
RULE = {"fallen": 30.0, "fall": 45.3, "fall_union": 55.6}
HYBRID_SEEDS = {
    "fallen": [34.7, 33.5, 34.7],
    "fall": [44.3, 45.8, 43.7],
    "fall_union": [57.9, 57.1, 56.4],
}


def load_cache_preds(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    y_true, y_pred = [], []
    for item in data.values():
        y_true.extend(item["labels"])
        y_pred.extend(np.argmax(item["logits"], axis=-1))
    return np.array(y_true), np.array(y_pred)


@torch.no_grad()
def load_v8_preds_all_seeds():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(RULE_CACHE, "rb") as f:
        data = pickle.load(f)

    y_true_all, y_pred_all = [], []
    for seed in SEEDS:
        ckpt = PROJECT_ROOT / "checkpoints" / f"learnable_conditioner_v8_stsyn_kl0.3_seed{seed}_best.pt"
        model = LearnableConditioner(
            feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
            context_window_seconds=15.0, residual=True, dropout=0.2,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        for item in data.values():
            feats = torch.tensor(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0).to(device)
            starts = torch.tensor([item["starts"]], dtype=torch.float).to(device)
            mask = torch.ones(1, feats.shape[1], dtype=torch.bool).to(device)
            logits = model(feats, starts, mask)
            preds = logits.argmax(dim=-1)[0].cpu().numpy()
            y_true_all.extend(item["labels"])
            y_pred_all.extend(preds)
    return np.array(y_true_all), np.array(y_pred_all)


def plot_confusion_4class(y_true, y_pred, title, ax):
    # 4 lớp: fall(1), lie_down(5), fallen(2), lying(6) -- giống thứ tự Figure 7 OmniFall
    idx = [1, 5, 2, 6]
    names = ["fall", "lie_down", "fallen", "lying"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    sub = cm[np.ix_(idx, idx)]
    sub_norm = sub / sub.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(sub_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            color = "white" if sub_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{sub_norm[i,j]*100:.1f}%\n(n={sub[i,j]})", ha="center", va="center",
                    color=color, fontsize=8)
    return im


def make_confusion_figure():
    y_true_b, y_pred_b = load_cache_preds(BASELINE_CACHE)
    y_true_h, y_pred_h = load_v8_preds_all_seeds()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = plot_confusion_4class(y_true_b, y_pred_b, "Baseline (St+Sy)", axes[0])
    im2 = plot_confusion_4class(y_true_h, y_pred_h, "Hybrid v8 (3 seed gộp)", axes[1])
    fig.colorbar(im2, ax=axes, fraction=0.046, pad=0.04, label="Tỷ lệ theo hàng (recall)")
    fig.suptitle("Confusion matrix 4 lớp (fall/lie_down/fallen/lying) — OF-ItW (OOPS) test", y=1.02)
    out_path = FIG_DIR / "confusion_matrix_4class.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")


def make_bar_chart():
    metrics = [("fallen", "Fallen-F1 (%)"), ("fall", "Fall-F1 (%)"), ("fall_union", "Fall/Fallen Union F1 (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (key, ylabel) in zip(axes, metrics):
        methods = ["Baseline", "Rule-based", "Hybrid v8"]
        means = [BASELINE[key], RULE[key], float(np.mean(HYBRID_SEEDS[key]))]
        stds = [0, 0, float(np.std(HYBRID_SEEDS[key], ddof=1))]
        colors = ["#9e9e9e", "#4a90d9", "#e07b39"]

        bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.7)
        # Overlay các điểm seed riêng lẻ cho Hybrid v8
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(HYBRID_SEEDS[key]))
        ax.scatter([2 + j for j in jitter], HYBRID_SEEDS[key], color="black", zorder=5, s=20, label="seed riêng lẻ")

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(means) + max(stds) + 10)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 1.5, f"{mean:.1f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=15)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("So sánh Baseline / Rule-based / Hybrid v8 (mean ± std qua 3 seed) — OF-ItW test", y=1.03)
    fig.tight_layout()
    out_path = FIG_DIR / "comparison_bar_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")


if __name__ == "__main__":
    make_confusion_figure()
    make_bar_chart()
