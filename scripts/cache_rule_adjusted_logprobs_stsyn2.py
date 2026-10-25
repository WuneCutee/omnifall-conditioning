"""
Giống cache_rule_adjusted_logprobs.py nhưng cho baseline St+Sy mới (checkpoint-7000,
outputs/2026-09-08/21-55-48) — áp RuleBasedConditioner lên logits đã cache trong
baseline_logits/{stsyn_train,stsyn_val,OOPS_val_stsyn,OOPS_test_stsyn}.pkl.

Usage:
    python -m scripts.cache_rule_adjusted_logprobs_stsyn
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

SPLITS = ["stsyn2_train", "stsyn2_val", "OOPS_val_stsyn2", "OOPS_test_stsyn2"]


def process(name: str):
    with open(CACHE_DIR / f"{name}.pkl", "rb") as f:
        data = pickle.load(f)

    conditioner = RuleBasedConditioner(boost_window_seconds=10.0, decay_halflife_seconds=4.0)
    result = {}

    for path, item in data.items():
        logits = torch.from_numpy(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0)
        starts = torch.tensor([item["starts"]], dtype=torch.float)
        mask = torch.ones(1, logits.shape[1], dtype=torch.bool)

        probs = F.softmax(logits, dim=-1)
        adjusted = conditioner.apply(probs.clone(), starts, mask)
        log_adjusted = torch.log(adjusted + EPS)[0].numpy()

        result[path] = {"logits": log_adjusted, "labels": item["labels"], "starts": item["starts"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"{name}: {len(result)} videos -> {OUT_DIR / f'{name}.pkl'}")


def main():
    for name in SPLITS:
        process(name)


if __name__ == "__main__":
    main()
