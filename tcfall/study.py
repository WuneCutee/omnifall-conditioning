"""Tiện ích dùng chung cho các bước thí nghiệm: nạp logits, tạo xác suất segment, áp conditioner."""
from __future__ import annotations

import pickle

import numpy as np
from scipy.special import softmax

from tcfall.config import NUM_CLASSES, OUTPUTS, DATA
from tcfall.conditioners import hmm_filter, hybrid_apply, prior_shift, rule_condition

CROP_SETS = {"center": [4], "tta3": [0, 4, 8], "tta5": [0, 2, 4, 6, 8], "tta9": list(range(9))}


def load_logits(variant: str, split: str) -> dict:
    with open(OUTPUTS / "logits" / f"{variant}_{split}.pkl", "rb") as f:
        return pickle.load(f)


def video_probs(item: dict, crops: str = "center", random_draw: int | None = None) -> np.ndarray:
    """Xác suất mỗi segment (T, C): trung bình softmax qua các crop thời gian (TTA) hoặc 1 lần
    rút ngẫu nhiên. Nếu item có khoá "spatial" (danh sách item crop trái/phải), lấy trung bình
    thêm qua các crop không gian (TTA không gian)."""
    if random_draw is not None:
        return softmax(item["random"][:, random_draw], -1)
    P = softmax(item["crops"][:, CROP_SETS[crops]], -1).mean(1)
    if "spatial" in item:
        P = np.mean([P] + [softmax(s["crops"][:, CROP_SETS[crops]], -1).mean(1) for s in item["spatial"]], 0)
    return P


def load_with_spatial(split: str) -> dict:
    """timefix (crop giữa) + crop không gian trái/phải ghép vào khoá "spatial"."""
    base = load_logits("timefix", split)
    extra = [load_logits(v, split) for v in ("timefix_start", "timefix_end")]
    return {k: {**v, "spatial": [e[k] for e in extra]} for k, v in base.items()}


def source_prior() -> np.ndarray:
    """Tần suất lớp của dữ liệu train baseline (St+Sy), lấy từ cache nhãn train cũ."""
    with open(DATA / "baseline_logits" / "stsyn_train.pkl", "rb") as f:
        d = pickle.load(f)
    y = np.concatenate([v["labels"] for v in d.values()])
    c = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    return c / c.sum()


def apply_pipeline(items: dict, cfg: dict, hybrid_model=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trả về [(y, pred)] theo video. cfg: crops, random_draw, prior(alpha, target), cond, tham số."""
    out = []
    for path in sorted(items):
        it = items[path]
        P = video_probs(it, cfg.get("crops", "center"), cfg.get("random_draw"))
        if cfg.get("prior_alpha", 0) > 0:
            P = prior_shift(P, cfg["src_prior"], cfg["tgt_prior"], cfg["prior_alpha"])
        cond = cfg.get("cond", "none")
        if cond in ("rule", "hybrid"):
            P = rule_condition(P, it["starts"], cfg.get("tau", 4.0), cfg.get("window", 10.0))
            if cond == "hybrid":
                P = hybrid_apply(hybrid_model, P, it["starts"])
        elif cond == "hmm":
            P = hmm_filter(P, cfg["A"], cfg["hmm_prior"], cfg.get("strength", 1.0))
        out.append((np.asarray(it["labels"]), P.argmax(1)))
    return out


def flat(per_video):
    return np.concatenate([y for y, _ in per_video]), np.concatenate([p for _, p in per_video])
