"""Đánh giá 1 checkpoint hybrid v5 (theo seed) trên OOPS test, in ra metrics đầy đủ.

Usage:
    python -m scripts.eval_v5_seed --ckpt checkpoints/learnable_conditioner_v5_stsyn_seed43_best.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner  # noqa: E402
from src.utils import compute_all_metrics, print_metrics_table  # noqa: E402

RULE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs" / "OOPS_test_stsyn.pkl"
NUM_CLASSES = 10


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LearnableConditioner(
        feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
        context_window_seconds=15.0, residual=True, dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
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

    m = compute_all_metrics(np.array(y_true), np.array(y_pred), fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
    print_metrics_table(m)


if __name__ == "__main__":
    main()
