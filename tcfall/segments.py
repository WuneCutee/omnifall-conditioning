"""Ánh xạ segment (start, end, giây) -> 18 chỉ số feature liên tiếp.

Tái tạo CHÍNH XÁC logic của FeatureVideoDataset (omnifall-experiments/data/feature_dataset.py),
chỉ thay phần ngẫu nhiên bằng tham số `u` trong [0, 1]:
  * segment dài hơn chuỗi 18 feature (2.27 s): vị trí bắt đầu = start + u * (dur - 2.27)
  * segment ngắn hơn: căn giữa segment trong chuỗi, dịch ±min(0.25·dur, lề/2) theo (2u - 1)
Code gốc rút u ~ U(0,1) mỗi lần đánh giá -> kết quả test dao động giữa các lần chạy.
Ở đây u được cố định: u=0.5 (1 lát giữa) hoặc u = linspace(0, 1, K) (K lát, TTA).
"""
from __future__ import annotations

import numpy as np

from tcfall.config import FEATURE_STRIDE, NUM_FEATURES, TARGET_FPS, WINDOW_FRAMES

WINDOW_DUR = WINDOW_FRAMES / TARGET_FPS                     # 2.133 s
SEQ_SPAN = (NUM_FEATURES - 1) * FEATURE_STRIDE              # khoảng cách tâm feature đầu-cuối


def _idx_at(t: float) -> int:
    return max(0, round((t - WINDOW_DUR / 2) / FEATURE_STRIDE))


def window_start(start: float, end: float, u: float, n_feat: int) -> int:
    dur = end - start
    if SEQ_SPAN <= dur:
        idx = _idx_at(start + u * (dur - SEQ_SPAN))
    else:
        margin = SEQ_SPAN - dur
        ideal = start - margin / 2
        jitter = min(0.25 * dur, margin / 2)
        idx = max(0, _idx_at(ideal + (2 * u - 1) * jitter))
    end_idx = idx + NUM_FEATURES
    if end_idx > n_feat:
        idx = max(0, n_feat - NUM_FEATURES)
    if idx >= n_feat:
        idx = max(0, n_feat - 1)
    return idx


def gather(feats: np.ndarray, idx: int) -> np.ndarray:
    """18 feature từ idx, lặp feature cuối nếu thiếu (giống code gốc)."""
    seq = feats[idx:idx + NUM_FEATURES]
    if len(seq) < NUM_FEATURES:
        seq = np.concatenate([seq, np.repeat(seq[-1:], NUM_FEATURES - len(seq), 0)], 0)
    return seq


def crop_positions(k: int) -> np.ndarray:
    return np.array([0.5]) if k == 1 else np.linspace(0.0, 1.0, k)
