"""
McNemar per-class (fallen, fall) cho Hybrid v8 vs Rule-based, cho từng seed riêng lẻ
(không phải ensemble) -- dùng để tổng hợp lại kết quả chính xác cho results_final.

Usage:
    python -m scripts.mcnemar_per_class_v8
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner  # noqa: E402

RULE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs" / "OOPS_test_stsyn.pkl"
NUM_CLASSES = 10
SEEDS = [42, 43, 44]


@torch.no_grad()
def load_preds(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LearnableConditioner(
        feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
        context_window_seconds=15.0, residual=True, dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    with open(RULE_CACHE, "rb") as f:
        data = pickle.load(f)

    y_true, y_rule, y_hybrid = [], [], []
    for item in data.values():
        feats = torch.tensor(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0).to(device)
        starts = torch.tensor([item["starts"]], dtype=torch.float).to(device)
        mask = torch.ones(1, feats.shape[1], dtype=torch.bool).to(device)
        logits = model(feats, starts, mask)
        y_hybrid.extend(logits.argmax(dim=-1)[0].cpu().numpy())
        y_rule.extend(feats.argmax(dim=-1)[0].cpu().numpy())
        y_true.extend(item["labels"])
    return np.array(y_true), np.array(y_rule), np.array(y_hybrid)


def mcnemar_for_class(y_true, y_rule, y_hybrid, class_id):
    mask = y_true == class_id
    n = int(mask.sum())
    rule_correct = (y_rule == class_id)[mask]
    hybrid_correct = (y_hybrid == class_id)[mask]
    both = int((rule_correct & hybrid_correct).sum())
    only_rule = int((rule_correct & ~hybrid_correct).sum())
    only_hybrid = int((~rule_correct & hybrid_correct).sum())
    neither = int((~rule_correct & ~hybrid_correct).sum())
    if only_rule + only_hybrid > 0:
        stat = (abs(only_rule - only_hybrid) - 1) ** 2 / (only_rule + only_hybrid)
        p = float(1 - chi2.cdf(stat, df=1))
    else:
        stat, p = 0.0, 1.0
    return {"n": n, "both": both, "only_rule": only_rule, "only_hybrid": only_hybrid,
            "neither": neither, "chi2": float(stat), "p_value": p}


def main():
    results = {}
    for seed in SEEDS:
        ckpt = PROJECT_ROOT / "checkpoints" / f"learnable_conditioner_v8_stsyn_kl0.3_seed{seed}_best.pt"
        y_true, y_rule, y_hybrid = load_preds(ckpt)
        fallen = mcnemar_for_class(y_true, y_rule, y_hybrid, class_id=2)
        fall = mcnemar_for_class(y_true, y_rule, y_hybrid, class_id=1)
        print(f"Seed {seed} -- fallen (n={fallen['n']}): chi2={fallen['chi2']:.3f} p={fallen['p_value']:.4f} "
              f"only_rule={fallen['only_rule']} only_hybrid={fallen['only_hybrid']}")
        print(f"Seed {seed} -- fall   (n={fall['n']}): chi2={fall['chi2']:.3f} p={fall['p_value']:.4f} "
              f"only_rule={fall['only_rule']} only_hybrid={fall['only_hybrid']}")
        results[f"seed{seed}"] = {"fallen": fallen, "fall": fall}

    out_path = PROJECT_ROOT / "data" / "omnifall" / "mcnemar_per_class_v8_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
