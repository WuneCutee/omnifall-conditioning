"""Metric và kiểm định thống kê dùng chung."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import balanced_accuracy_score, f1_score

from tcfall.config import FALL, FALLEN


def core_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    yu, pu = np.isin(y, [FALL, FALLEN]), np.isin(p, [FALL, FALLEN])
    return {
        "fallen_f1": 100 * f1_score(y == FALLEN, p == FALLEN, zero_division=0),
        "fall_f1": 100 * f1_score(y == FALL, p == FALL, zero_division=0),
        "union_f1": 100 * f1_score(yu, pu, zero_division=0),
        "union_se": 100 * (yu & pu).sum() / max(yu.sum(), 1),
        "union_sp": 100 * (~yu & ~pu).sum() / max((~yu).sum(), 1),
        "bacc": 100 * balanced_accuracy_score(y, p),
        "macro_f1": 100 * f1_score(y, p, average="macro", zero_division=0),
    }


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    only_a = int((a_correct & ~b_correct).sum()); only_b = int((~a_correct & b_correct).sum())
    if only_a + only_b == 0:
        return {"only_a": 0, "only_b": 0, "chi2": 0.0, "p": 1.0}
    stat = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
    return {"only_a": only_a, "only_b": only_b, "chi2": float(stat), "p": float(1 - chi2.cdf(stat, 1))}


def video_bootstrap(per_video: list[tuple[np.ndarray, np.ndarray, np.ndarray]], metric: str,
                    n_boot: int = 10000, seed: int = 0) -> dict:
    """CI 95% cho hiệu metric(B) - metric(A), resample theo VIDEO. per_video = [(y, pA, pB)]."""
    pos = {"fallen_f1": [FALLEN], "fall_f1": [FALL], "union_f1": [FALL, FALLEN]}[metric]
    # Đếm TP/FP/FN theo từng video 1 lần, mỗi vòng bootstrap chỉ cần cộng -> nhanh
    cnt = np.array([[(np.isin(y, pos) & np.isin(p, pos)).sum(), (~np.isin(y, pos) & np.isin(p, pos)).sum(),
                     (np.isin(y, pos) & ~np.isin(p, pos)).sum()] for y, a, b in per_video for p in (a, b)]
                   ).reshape(len(per_video), 2, 3).astype(float)
    rng = np.random.default_rng(seed)
    n = len(per_video)
    idx = rng.integers(0, n, (n_boot, n))
    tot = cnt[idx].sum(1)  # (n_boot, 2, 3)
    f1 = 100 * 2 * tot[..., 0] / np.maximum(2 * tot[..., 0] + tot[..., 1] + tot[..., 2], 1)
    diffs = f1[:, 1] - f1[:, 0]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi), "p_le0": float((diffs <= 0).mean())}
