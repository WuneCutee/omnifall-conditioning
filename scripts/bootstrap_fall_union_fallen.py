"""
Kiểm định thống kê cho Fall∪Fallen F1 (Hybrid v8 vs Rule-based), cho từng seed.

F1 không phải accuracy nhị phân đơn giản (phụ thuộc TP/FP/FN cùng lúc), nên McNemar không áp
dụng trực tiếp được cho F1 — dùng bootstrap trên đơn vị VIDEO (không phải segment độc lập, vì
các segment trong cùng 1 video không độc lập — bootstrap segment riêng lẻ sẽ đánh giá thấp
variance thật). Resample video có hoàn lại 10,000 lần, tính lại Fall∪Fallen F1 (rule & hybrid)
mỗi lần trên đúng tập segment của các video được chọn, lấy CI 95% cho hiệu (Hybrid - Rule).

Kèm theo: McNemar trên "union-accuracy" (đúng/sai cho việc phân loại đúng vào {fall,fallen}
hay không) để tam giác hoá kết quả.

Usage:
    python -m scripts.bootstrap_fall_union_fallen
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner  # noqa: E402

RULE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs" / "OOPS_test_stsyn.pkl"
NUM_CLASSES = 10
N_BOOTSTRAP = 10000
SEEDS = [42, 43, 44]


def load_per_video_preds(ckpt_path):
    """Trả về list các (y_true, y_pred_rule, y_pred_hybrid) theo TỪNG VIDEO (giữ nguyên nhóm)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LearnableConditioner(
        feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
        context_window_seconds=15.0, residual=True, dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    with open(RULE_CACHE, "rb") as f:
        data = pickle.load(f)

    per_video = []
    with torch.no_grad():
        for item in data.values():
            feats = torch.tensor(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0).to(device)
            starts = torch.tensor([item["starts"]], dtype=torch.float).to(device)
            mask = torch.ones(1, feats.shape[1], dtype=torch.bool).to(device)
            logits = model(feats, starts, mask)
            pred_hybrid = logits.argmax(dim=-1)[0].cpu().numpy()
            pred_rule = feats.argmax(dim=-1)[0].cpu().numpy()
            y_true = np.array(item["labels"])
            per_video.append((y_true, pred_rule, pred_hybrid))
    return per_video


def union_f1(y_true, y_pred):
    yt = np.isin(y_true, [1, 2]).astype(int)
    yp = np.isin(y_pred, [1, 2]).astype(int)
    return f1_score(yt, yp, zero_division=0)


def bootstrap_ci(per_video, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    n_videos = len(per_video)
    diffs = []
    rule_f1s, hybrid_f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_videos, size=n_videos)
        y_true = np.concatenate([per_video[i][0] for i in idx])
        y_rule = np.concatenate([per_video[i][1] for i in idx])
        y_hybrid = np.concatenate([per_video[i][2] for i in idx])
        f1_rule = union_f1(y_true, y_rule)
        f1_hybrid = union_f1(y_true, y_hybrid)
        diffs.append(f1_hybrid - f1_rule)
        rule_f1s.append(f1_rule)
        hybrid_f1s.append(f1_hybrid)
    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value_one_sided = float((diffs <= 0).mean())  # tỷ lệ bootstrap mà Hybrid <= Rule
    return {
        "mean_diff": float(diffs.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_le_zero": p_value_one_sided,
        "rule_f1_point": float(union_f1(
            np.concatenate([v[0] for v in per_video]),
            np.concatenate([v[1] for v in per_video]))),
        "hybrid_f1_point": float(union_f1(
            np.concatenate([v[0] for v in per_video]),
            np.concatenate([v[2] for v in per_video]))),
    }


def mcnemar_union_accuracy(per_video):
    y_true = np.concatenate([v[0] for v in per_video])
    y_rule = np.concatenate([v[1] for v in per_video])
    y_hybrid = np.concatenate([v[2] for v in per_video])
    true_union = np.isin(y_true, [1, 2])
    rule_correct = np.isin(y_rule, [1, 2]) == true_union
    hybrid_correct = np.isin(y_hybrid, [1, 2]) == true_union

    only_rule = int((rule_correct & ~hybrid_correct).sum())
    only_hybrid = int((~rule_correct & hybrid_correct).sum())
    both = int((rule_correct & hybrid_correct).sum())
    neither = int((~rule_correct & ~hybrid_correct).sum())

    if only_rule + only_hybrid > 0:
        stat = (abs(only_rule - only_hybrid) - 1) ** 2 / (only_rule + only_hybrid)
        p = float(1 - chi2.cdf(stat, df=1))
    else:
        stat, p = 0.0, 1.0
    return {"both": both, "only_rule": only_rule, "only_hybrid": only_hybrid,
            "neither": neither, "chi2": float(stat), "p_value": p}


def main():
    results = {}
    for seed in SEEDS:
        ckpt = PROJECT_ROOT / "checkpoints" / f"learnable_conditioner_v8_stsyn_kl0.3_seed{seed}_best.pt"
        print(f"\n{'='*80}\nSeed {seed}\n{'='*80}")
        per_video = load_per_video_preds(ckpt)

        boot = bootstrap_ci(per_video, seed=seed)
        print(f"Fall∪Fallen F1 điểm: Rule-based={boot['rule_f1_point']*100:.2f}%, "
              f"Hybrid v8={boot['hybrid_f1_point']*100:.2f}%")
        print(f"Bootstrap (n={N_BOOTSTRAP}, resample theo VIDEO): "
              f"mean diff={boot['mean_diff']*100:+.2f} điểm, "
              f"95% CI=[{boot['ci95_low']*100:+.2f}, {boot['ci95_high']*100:+.2f}], "
              f"P(Hybrid<=Rule)={boot['p_le_zero']:.4f}")
        significant = boot["ci95_low"] > 0
        print(f"  => {'CÓ Ý NGHĨA THỐNG KÊ (CI không chứa 0, Hybrid > Rule)' if significant else 'KHÔNG loại trừ được 0 khỏi CI'}")

        mcn = mcnemar_union_accuracy(per_video)
        print(f"McNemar (union-accuracy): both={mcn['both']} only_rule={mcn['only_rule']} "
              f"only_hybrid={mcn['only_hybrid']} neither={mcn['neither']} "
              f"chi2={mcn['chi2']:.3f} p={mcn['p_value']:.4f}")

        results[f"seed{seed}"] = {"bootstrap": boot, "mcnemar_union_accuracy": mcn}

    out_path = PROJECT_ROOT / "data" / "omnifall" / "bootstrap_fall_union_fallen_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu kết quả đầy đủ: {out_path}")


if __name__ == "__main__":
    main()
