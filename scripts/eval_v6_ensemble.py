"""
Bước 3 — Ensemble 3 seed của Hybrid v6 (average probs), so với rule-based (đã sửa lỗi) +
McNemar test. Không cần train lại — dùng 3 checkpoint đã có.

Usage:
    python -m scripts.eval_v6_ensemble
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner  # noqa: E402
from src.utils import compute_all_metrics, print_metrics_table  # noqa: E402

RULE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs" / "OOPS_test_stsyn.pkl"
CKPTS = [
    PROJECT_ROOT / "checkpoints" / "learnable_conditioner_v6_stsyn_seed42_best.pt",
    PROJECT_ROOT / "checkpoints" / "learnable_conditioner_v6_stsyn_seed43_best.pt",
    PROJECT_ROOT / "checkpoints" / "learnable_conditioner_v6_stsyn_seed44_best.pt",
]
NUM_CLASSES = 10


def load_models(device):
    models = []
    for ckpt in CKPTS:
        m = LearnableConditioner(
            feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
            context_window_seconds=15.0, residual=True, dropout=0.2,
        ).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)

    with open(RULE_CACHE, "rb") as f:
        data = pickle.load(f)

    y_true, y_pred_ens, y_pred_rule = [], [], []
    for item in data.values():
        feats = torch.tensor(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0).to(device)
        starts = torch.tensor([item["starts"]], dtype=torch.float).to(device)
        mask = torch.ones(1, feats.shape[1], dtype=torch.bool).to(device)

        probs_sum = None
        for m in models:
            logits = m(feats, starts, mask)
            probs = F.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs_avg = probs_sum / len(models)
        preds_ens = probs_avg.argmax(dim=-1)[0].cpu().numpy()
        preds_rule = feats.argmax(dim=-1)[0].cpu().numpy()  # rule-based (input logits chính là rule-based)

        y_true.extend(item["labels"])
        y_pred_ens.extend(preds_ens)
        y_pred_rule.extend(preds_rule)

    y_true = np.array(y_true)
    y_pred_ens = np.array(y_pred_ens)
    y_pred_rule = np.array(y_pred_rule)

    print("=== RULE-BASED (đã sửa lỗi) ===")
    m_rule = compute_all_metrics(y_true, y_pred_rule, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
    print_metrics_table(m_rule)

    print("\n=== ENSEMBLE (3 seed Hybrid v6, average probs) ===")
    m_ens = compute_all_metrics(y_true, y_pred_ens, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
    print_metrics_table(m_ens)

    delta = (m_ens["fallen"]["f1"] - m_rule["fallen"]["f1"]) * 100
    print(f"\n>>> Thay đổi Fallen-F1 (Ensemble vs Rule-based): {delta:+.2f} điểm")

    # McNemar test cho true=fallen
    mask_fallen = y_true == 2
    rule_correct = (y_pred_rule == 2)[mask_fallen]
    ens_correct = (y_pred_ens == 2)[mask_fallen]
    both = int((rule_correct & ens_correct).sum())
    only_rule = int((rule_correct & ~ens_correct).sum())
    only_ens = int((~rule_correct & ens_correct).sum())
    neither = int((~rule_correct & ~ens_correct).sum())
    print(f"\nMcNemar (true=fallen, n={mask_fallen.sum()}): both={both} only_rule={only_rule} only_ens={only_ens} neither={neither}")
    if only_rule + only_ens > 0:
        stat = (abs(only_rule - only_ens) - 1) ** 2 / (only_rule + only_ens)
        p = 1 - chi2.cdf(stat, df=1)
        print(f"McNemar chi2={stat:.3f} p={p:.4f} "
              f"{'=> CÓ ý nghĩa thống kê' if p < 0.05 else '=> KHÔNG có ý nghĩa thống kê'}"
              f"{' (nghiêng về Ensemble)' if only_ens > only_rule else ' (nghiêng về Rule-based)' if only_rule > only_ens else ''}")
    else:
        print("Không đủ cặp lệch để chạy McNemar.")


if __name__ == "__main__":
    main()
