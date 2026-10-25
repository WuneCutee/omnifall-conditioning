"""
Bước 4 — Ablation đầy đủ: tổng hợp tất cả biến thể trên CÙNG 1 tập test (OOPS), tính thêm
per-class F1 cho fallen/lying/lie_down/fall (trọng tâm thật của nghiên cứu là phân biệt
fallen<->lying, không chỉ Fall-F1) và confusion giữa 2 lớp đó.

Usage:
    python -m scripts.run_final_ablation
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner  # noqa: E402
from src.utils import compute_all_metrics  # noqa: E402

BASELINE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "baseline_logits" / "OOPS_test.pkl"
RULE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs" / "OOPS_test.pkl"
V4_CKPT = PROJECT_ROOT / "checkpoints" / "learnable_conditioner_v4_best.pt"

CLASS_NAMES = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
               "stand_up", "standing", "other"]
NUM_CLASSES = 10


def load_cache_preds(pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    y_true, y_pred = [], []
    for item in data.values():
        y_true.extend(item["labels"])
        y_pred.extend(np.argmax(item["logits"], axis=-1))
    return np.array(y_true), np.array(y_pred)


@torch.no_grad()
def load_v4_preds() -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LearnableConditioner(
        feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
        context_window_seconds=15.0, residual=True, dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(V4_CKPT, map_location=device))
    model.eval()

    with open(RULE_CACHE, "rb") as f:
        data = pickle.load(f)

    y_true, y_pred = [], []
    for item in data.values():
        feats = torch.tensor(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0).to(device)
        starts = torch.tensor([item["starts"]], dtype=torch.float).to(device)
        mask = torch.ones(1, feats.shape[1], dtype=torch.bool).to(device)
        logits = model(feats, starts, mask)
        preds = logits.argmax(dim=-1)[0].cpu().numpy()
        y_true.extend(item["labels"])
        y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)


def per_class_f1(y_true, y_pred) -> dict:
    f1s = f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    return {CLASS_NAMES[i]: f1s[i] for i in range(NUM_CLASSES)}


def print_row(name, y_true, y_pred):
    m = compute_all_metrics(y_true, y_pred, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
    pc = per_class_f1(y_true, y_pred)
    print(f"{name:<28} "
          f"Fallen={pc['fallen']*100:5.1f}% "
          f"Fall={pc['fall']*100:5.1f}% "
          f"Lying={pc['lying']*100:5.1f}% "
          f"LieDown={pc['lie_down']*100:5.1f}% "
          f"Fall∪Fallen={m['fall_union_fallen']['f1']*100:5.1f}% "
          f"BalAcc={m['balanced_accuracy']*100:5.1f}% "
          f"MacroF1={m['macro_f1']*100:5.1f}%")
    return m, pc


def fallen_lying_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fallen, lying = 2, 6
    print(f"  [{name}] Nhầm lẫn fallen<->lying: "
          f"fallen->lying={cm[fallen, lying]}/{cm[fallen].sum()} "
          f"({100*cm[fallen, lying]/max(cm[fallen].sum(),1):.1f}%), "
          f"lying->fallen={cm[lying, fallen]}/{cm[lying].sum()} "
          f"({100*cm[lying, fallen]/max(cm[lying].sum(),1):.1f}%)")


def main():
    print("=" * 140)
    print("ABLATION ĐẦY ĐỦ — OF-Syn train -> OF-ItW (OOPS) test, cùng checkpoint baseline (checkpoint-3000)")
    print("=" * 140)

    y_true_base, y_pred_base = load_cache_preds(BASELINE_CACHE)
    y_true_rule, y_pred_rule = load_cache_preds(RULE_CACHE)
    y_true_v4, y_pred_v4 = load_v4_preds()

    print(f"\nSố segment test: {len(y_true_base)}\n")

    print_row("1. Baseline (no conditioning)", y_true_base, y_pred_base)
    print_row("2. Rule-based (post-hoc)", y_true_rule, y_pred_rule)
    print_row("3. Hybrid v4 (learnable, warm-start rule)", y_true_v4, y_pred_v4)

    print("\n--- Nhầm lẫn Fallen <-> Lying (trọng tâm nghiên cứu) ---")
    fallen_lying_confusion(y_true_base, y_pred_base, "Baseline")
    fallen_lying_confusion(y_true_rule, y_pred_rule, "Rule-based")
    fallen_lying_confusion(y_true_v4, y_pred_v4, "Hybrid v4")

    print("\n--- Đã tổng hợp kèm số liệu learnable v1/v2/v3 từ log trước đó (không tính lại) ---")
    print("v1 (from-scratch, raw feat):     Fallen=28.6% Fall=42.1% Fall∪Fallen=50.6% BalAcc=24.7% MacroF1=22.4%")
    print("v2 (residual/baseline, syn-val): Fallen=28.6% Fall=34.8% Fall∪Fallen=42.3% BalAcc=21.2% MacroF1=20.3%")
    print("v3 (+ OOPS-val, balanced, drop): Fallen=29.5% Fall=34.9% Fall∪Fallen=45.4% BalAcc=20.6% MacroF1=19.8%")


if __name__ == "__main__":
    main()
