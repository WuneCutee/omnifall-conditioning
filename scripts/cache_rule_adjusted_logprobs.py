"""
Bước 3v4 — tính lại cache logits, nhưng lần này áp RuleBasedConditioner LÊN TRÊN baseline
probs trước (giống Bước 2, đã cho +12.9 điểm Fallen-F1), rồi lưu log(probs đã điều chỉnh)
làm input mới cho LearnableConditioner residual. Ý tưởng: model học chỉ cần SỬA THÊM trên
nền rule-based đã tốt, thay vì phải tự học lại từ baseline thô (khó hơn nhiều với lượng
dữ liệu multi-segment hạn chế).

Usage:
    python -m scripts.cache_rule_adjusted_logprobs
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import RuleBasedConditioner  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "data" / "omnifall" / "baseline_logits"
OUT_DIR = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs"
EPS = 1e-6


def process(name: str):
    with open(CACHE_DIR / f"{name}.pkl", "rb") as f:
        data = pickle.load(f)

    conditioner = RuleBasedConditioner(boost_window_seconds=10.0, decay_halflife_seconds=4.0)
    result = {}

    for path, item in data.items():
        logits = torch.from_numpy(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0)  # (1, seq, 10)
        starts = torch.tensor([item["starts"]], dtype=torch.float)  # (1, seq)
        mask = torch.ones(1, logits.shape[1], dtype=torch.bool)

        probs = F.softmax(logits, dim=-1)
        adjusted = conditioner.apply(probs.clone(), starts, mask)  # (1, seq, 10)
        log_adjusted = torch.log(adjusted + EPS)[0].numpy()  # (seq, 10)

        result[path] = {"logits": log_adjusted, "labels": item["labels"], "starts": item["starts"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"{name}: {len(result)} videos -> {OUT_DIR / f'{name}.pkl'}")


def main():
    for name in ["of-syn_train", "of-syn_val", "OOPS_val", "OOPS_test"]:
        process(name)


if __name__ == "__main__":
    main()
