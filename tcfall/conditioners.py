"""Các bộ điều kiện hoá theo thời gian (temporal conditioners) hoạt động trên chuỗi xác suất
segment của 1 video. Tất cả đều NHÂN QUẢ (chỉ dùng quá khứ) để dùng được realtime.

  * rule_condition  — Transition-Conditioning dựa luật (bài báo, Eq. 1-2), vector hoá.
  * hmm_filter      — lọc tiến HMM (forward filtering) với ma trận chuyển tiếp ước lượng từ
                      nhãn train; là đối chứng "chuẩn sách giáo khoa" cho cấu trúc hành động-trạng thái.
  * prior_shift     — hiệu chỉnh lệch phân bố lớp (label shift) giữa nguồn train và miền đích.
  * hybrid (LearnableConditioner) nạp từ checkpoint, xem load_hybrid().
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from tcfall.config import CKPT_DIR, NUM_CLASSES, ROOT, TRANSIENT_TO_STATIC

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.conditioning import LearnableConditioner  # noqa: E402


def rule_condition(probs: np.ndarray, starts: np.ndarray, tau: float = 4.0, window: float = 10.0) -> np.ndarray:
    """probs (T, C) -> probs đã điều kiện hoá. Giống hệt RuleBasedConditioner (đã sửa lỗi):
    không boost segment đang được dự đoán là 1 hành động chuyển tiếp; Δt=0 không được boost."""
    out = probs.copy()
    last: dict[int, float] = {}
    pred = probs.argmax(1)
    for t in range(len(probs)):
        if pred[t] not in TRANSIENT_TO_STATIC:
            for s, t0 in list(last.items()):
                dt = starts[t] - t0
                if dt > window:
                    del last[s]
                    continue
                w = 0.5 ** (dt / tau)
                prior = np.zeros(probs.shape[1]); prior[s] = 1.0
                out[t] = (1 - w) * out[t] + w * prior
        else:
            last[TRANSIENT_TO_STATIC[int(pred[t])]] = starts[t]
    return out / out.sum(1, keepdims=True).clip(min=1e-8)


def estimate_transitions(counts: np.ndarray, self_loop: float, smoothing: float = 1.0) -> np.ndarray:
    """Ma trận chuyển tiếp HMM từ ma trận đếm chuyển nhãn giữa 2 segment liên tiếp.
    self_loop: xác suất giữ nguyên lớp (segment liên tiếp trong nhãn gốc luôn khác lớp,
    nhưng dự đoán thì có thể lặp lại), phần còn lại chia theo tần suất chuyển tiếp."""
    off = counts.astype(float) + smoothing
    np.fill_diagonal(off, 0.0)
    off = off / off.sum(1, keepdims=True)
    return self_loop * np.eye(NUM_CLASSES) + (1 - self_loop) * off


def transitions_from_counts(counts: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Ma trận chuyển tiếp chuẩn hoá theo hàng, GIỮ đường chéo (dùng cho đếm lấy mẫu theo thời gian)."""
    M = counts.astype(float) + smoothing
    return M / M.sum(1, keepdims=True)


def sharpen_self_loop(A: np.ndarray, gamma: float) -> np.ndarray:
    """Điều chỉnh độ 'dính' của trạng thái: gamma>1 tăng self-loop, <1 giảm, rồi chuẩn hoá lại."""
    B = A.copy()
    np.fill_diagonal(B, np.diag(A) ** (1.0 / gamma))
    return B / B.sum(1, keepdims=True)


def hmm_filter(probs: np.ndarray, A: np.ndarray, class_prior: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Lọc tiến: alpha_t ∝ (A^T alpha_{t-1}) * (p_t / prior)^strength. Nhân quả, O(T·C²)."""
    lik = (probs / class_prior[None]) ** strength
    alpha = np.zeros_like(probs)
    a = np.full(probs.shape[1], 1.0 / probs.shape[1])
    for t in range(len(probs)):
        pred = a @ A if t > 0 else a
        a = pred * lik[t]
        a = a / a.sum().clip(min=1e-12)
        alpha[t] = a
    return alpha


def prior_shift(probs: np.ndarray, source_prior: np.ndarray, target_prior: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Hiệu chỉnh label shift (Saerens et al., 2002): p'(y|x) ∝ p(y|x) · (π_t / π_s)^alpha."""
    w = (target_prior / source_prior) ** alpha
    q = probs * w[None]
    return q / q.sum(1, keepdims=True)


def em_target_prior(probs: np.ndarray, source_prior: np.ndarray, iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Ước lượng prior miền đích KHÔNG cần nhãn bằng EM (Saerens et al., 2002)."""
    pi = source_prior.copy()
    for _ in range(iters):
        q = prior_shift(probs, source_prior, pi)
        new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


def load_hybrid(path, device: str = "cpu") -> LearnableConditioner:
    m = LearnableConditioner(feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
                             context_window_seconds=15.0, residual=True, dropout=0.2)
    m.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    return m.to(device).eval()


@torch.no_grad()
def hybrid_apply(model: LearnableConditioner, rule_probs: np.ndarray, starts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    dev = next(model.parameters()).device
    x = torch.tensor(np.log(rule_probs + eps), dtype=torch.float32, device=dev)[None]
    s = torch.tensor(starts, dtype=torch.float32, device=dev)[None]
    m = torch.ones(1, len(starts), dtype=torch.bool, device=dev)
    return torch.softmax(model(x, s, m), -1)[0].cpu().numpy()


DEFAULT_HYBRID = CKPT_DIR / "learnable_conditioner_v8_stsyn_kl0.3_seed42_best.pt"
