"""Nạp nhãn segment, gán split, làm sạch có ghi nhận (audit trail) và tạo chuỗi segment theo video."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tcfall.config import LABEL_FILES, LABELS_RAW, NUM_CLASSES, SPLITS

SPLIT_NAMES = ("train", "val", "test")


def load_labels(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(LABELS_RAW / LABEL_FILES[dataset])
    df["dataset"] = dataset
    df["dur"] = df["end"] - df["start"]
    return df


def load_split(dataset: str, protocol: str = "cs") -> dict[str, str]:
    """path -> train/val/test. OF-Syn dùng split 'random' (theo protocol của OmniFall)."""
    base = SPLITS / "syn" / "random" if dataset == "of-syn" else SPLITS / protocol / dataset
    out = {}
    for s in SPLIT_NAMES:
        f = base / f"{s}.csv"
        if f.exists():
            for p in pd.read_csv(f)["path"]:
                if p in out and out[p] != s:
                    out[p] = "CONFLICT"
                else:
                    out[p] = s
    return out


@dataclass
class CleanReport:
    dataset: str
    steps: list[tuple[str, int]] = field(default_factory=list)

    def add(self, name: str, n_removed: int):
        self.steps.append((name, int(n_removed)))


def clean_labels(df: pd.DataFrame, min_dur: float = 0.0, keep_extended: bool = False
                 ) -> tuple[pd.DataFrame, CleanReport]:
    """Các bước lọc có thứ tự; mỗi bước ghi số dòng bị loại.

    min_dur mặc định 0: KHÔNG loại segment ngắn khi đánh giá (giữ nguyên test set chuẩn),
    chỉ dùng min_dur > 0 cho dữ liệu huấn luyện.
    """
    rep = CleanReport(df["dataset"].iloc[0] if len(df) else "?")
    n = len(df); df = df.dropna(subset=["path", "label", "start", "end"]); rep.add("thiếu trường", n - len(df))
    n = len(df); df = df[df["end"] > df["start"]]; rep.add("end <= start", n - len(df))
    n = len(df); df = df.drop_duplicates(subset=["path", "start", "end", "label"]); rep.add("trùng lặp", n - len(df))
    if not keep_extended:
        n = len(df); df = df[(df["label"] >= 0) & (df["label"] < NUM_CLASSES)]; rep.add("lớp mở rộng (10-15)", n - len(df))
    if min_dur > 0:
        n = len(df); df = df[df["dur"] >= min_dur]; rep.add(f"ngắn hơn {min_dur}s", n - len(df))
    return df.sort_values(["path", "start"]).reset_index(drop=True), rep


def segment_sequences(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """path -> các segment của video đó theo thứ tự thời gian."""
    return {p: g.sort_values("start").reset_index(drop=True) for p, g in df.groupby("path", sort=False)}


def transition_counts_timesampled(df: pd.DataFrame, hop: float) -> np.ndarray:
    """Đếm chuyển tiếp giữa nhãn lấy mẫu mỗi `hop` giây (không dùng ranh giới segment) —
    đúng với cách pseudo-segment được tạo khi chạy streaming, nên có self-loop thực tế."""
    M = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for _, g in df.sort_values(["path", "start"]).groupby("path", sort=False):
        st, en, lab = g["start"].to_numpy(), g["end"].to_numpy(), g["label"].to_numpy()
        times = np.arange(st.min(), en.max(), hop)
        idx = np.searchsorted(st, times, side="right") - 1
        ok = (idx >= 0) & (times < en[np.clip(idx, 0, None)])
        seq = lab[idx[ok]]
        seq = seq[seq < NUM_CLASSES]
        np.add.at(M, (seq[:-1], seq[1:]), 1)
    return M


def transition_counts(df: pd.DataFrame) -> np.ndarray:
    """Ma trận đếm chuyển tiếp nhãn giữa 2 segment liên tiếp trong cùng video."""
    M = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for _, g in df.sort_values(["path", "start"]).groupby("path", sort=False):
        lab = g["label"].to_numpy()
        lab = lab[lab < NUM_CLASSES]
        for a, b in zip(lab[:-1], lab[1:]):
            M[a, b] += 1
    return M
