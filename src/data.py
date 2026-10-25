"""
Dataset loader cho OmniFall — dùng `datasets.load_dataset("simplexsigil2/omnifall", ...)`
chính thức (đọc từ parquet local trong data/omnifall/, không cần mạng nếu đã tải sẵn).

Format thật (xem data/omnifall/STRUCTURE.md), mỗi segment có cột:
  path, label, start, end, subject, cam, dataset
  - start/end: giây (float), KHÔNG phải frame index.
  - path: tương đối trong dataset, không đuôi file — video ở {root}/{dataset}/video/{path}.mp4
  - path chỉ duy nhất TRONG PHẠM VI 1 dataset -> luôn khoá theo cặp (dataset, path).

Feature đã trích xuất (xem features.py) là 1 mảng (T, D) cho mỗi (dataset, path), lấy mẫu
đều theo `fps`/`stride_seconds` cấu hình — KHÔNG map 1-1 với frame gốc của video. Việc đổi
start/end (giây) -> chỉ số trong mảng feature dùng chung công thức với features.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset

HF_REPO = "simplexsigil2/omnifall"

# 16-class taxonomy — khớp với data/omnifall/labels/label2id.csv thật (đã verify).
CLASS_NAMES = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other", "kneel_down", "kneeling",
    "squat_down", "squatting", "crawl", "jump",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Hành động chuyển tiếp (transient) -> trạng thái tĩnh (static) tương ứng theo sau,
# dùng cho module conditioning (xem conditioning.py).
TRANSIENT_TO_STATIC = {
    CLASS_TO_ID["fall"]: CLASS_TO_ID["fallen"],
    CLASS_TO_ID["sit_down"]: CLASS_TO_ID["sitting"],
    CLASS_TO_ID["lie_down"]: CLASS_TO_ID["lying"],
    CLASS_TO_ID["stand_up"]: CLASS_TO_ID["standing"],
}


def get_video_path(root_dir: str | Path, dataset: str, path: str) -> Path:
    """{root}/{dataset}/video/{path}.mp4 — quy tắc resolve video thật, xem STRUCTURE.md."""
    return Path(root_dir) / dataset / "video" / f"{path}.mp4"


def get_feature_path(feature_cache_dir: str | Path, dataset: str, path: str) -> Path:
    """Khoá cache feature theo (dataset, path) vì path chỉ duy nhất trong 1 dataset."""
    return Path(feature_cache_dir) / dataset / f"{path}.npy"


def stride_frames_from_seconds(fps: float, stride_seconds: float) -> int:
    """Cùng công thức với features.py: bước trượt (số resampled-frame) giữa 2 feature liên tiếp."""
    return max(int(round(stride_seconds * fps)), 1)


def time_to_feature_index(t: float, fps: float, stride_frames: int) -> int:
    """Giây -> chỉ số trong mảng feature. Feature index i ứng với cửa sổ bắt đầu tại
    resampled-frame (i * stride_frames), tức thời điểm (i * stride_frames / fps) giây."""
    return int(round(t * fps / stride_frames))


def load_split(config_name: str, split: str, keep_datasets: set[str] | None = None) -> HFDataset:
    """Load 1 split của 1 config OmniFall. Lọc theo cột `dataset` nếu `keep_datasets` được cho."""
    ds = load_dataset(HF_REPO, config_name)[split]
    if keep_datasets is not None:
        ds = ds.filter(lambda ex: ex["dataset"] in keep_datasets)
    return ds


def load_sources(sources: list[tuple[str, str]], keep_datasets: set[str] | None = None) -> HFDataset:
    """Load và nối nhiều (config_name, split) lại làm 1 — dùng để gộp nhiều nguồn train
    (vd sau này thêm OF-Staged khi có video thật: [("of-syn","train"), ("of-sta-cs","train")])."""
    parts = [load_split(cfg, split, keep_datasets) for cfg, split in sources]
    if len(parts) == 1:
        return parts[0]
    # Chỉ giữ các cột chung (of-syn có thêm 12 cột demographic mà of-sta-cs không có).
    common_cols = set(parts[0].column_names)
    for p in parts[1:]:
        common_cols &= set(p.column_names)
    parts = [p.remove_columns([c for c in p.column_names if c not in common_cols]) for p in parts]
    return concatenate_datasets(parts)


class FeatureCache:
    """LRU cache nhỏ giữ vài video gần nhất trong RAM — nhiều segment liên tiếp trong dataset
    thường thuộc cùng 1 video nên tránh phải đọc lại .npy từ đĩa mỗi lần."""

    def __init__(self, feature_cache_dir: str | Path, max_cached: int = 8):
        self.feature_cache_dir = Path(feature_cache_dir)
        self.max_cached = max_cached
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._order: list[tuple[str, str]] = []

    def get(self, dataset: str, path: str) -> np.ndarray:
        key = (dataset, path)
        if key in self._cache:
            return self._cache[key]
        feat_path = get_feature_path(self.feature_cache_dir, dataset, path)
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Chưa có feature cho (dataset={dataset}, path={path}) tại {feat_path}. "
                f"Chạy `python -m src.features` trước."
            )
        feat = np.load(feat_path)
        self._cache[key] = feat
        self._order.append(key)
        if len(self._order) > self.max_cached:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        return feat


def _segment_feature_slice(
    feats: np.ndarray, start: float, end: float, fps: float, stride_frames: int
) -> np.ndarray:
    i0 = time_to_feature_index(start, fps, stride_frames)
    i1 = time_to_feature_index(end, fps, stride_frames)
    i0 = max(0, min(i0, len(feats) - 1))
    i1 = max(i0 + 1, min(i1, len(feats)))
    return feats[i0:i1].mean(axis=0)


class OmniFallDataset(Dataset):
    """Trả về (feature, label, meta) cho 1 segment, mean-pool feature đã trích xuất sẵn
    trong khoảng [start, end) giây của segment đó."""

    def __init__(
        self,
        sources: list[tuple[str, str]],
        feature_cache_dir: str,
        fps: float,
        stride_seconds: float,
        keep_datasets: set[str] | None = None,
    ):
        self.rows = load_sources(sources, keep_datasets)
        self.feature_cache_dir = feature_cache_dir
        self.fps = fps
        self.stride_frames = stride_frames_from_seconds(fps, stride_seconds)
        self._features = FeatureCache(feature_cache_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        feats = self._features.get(row["dataset"], row["path"])
        feat = _segment_feature_slice(feats, row["start"], row["end"], self.fps, self.stride_frames)

        return {
            "feature": torch.from_numpy(np.asarray(feat)).float(),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "dataset": row["dataset"],
            "path": row["path"],
            "start": row["start"],
            "end": row["end"],
        }


class VideoLevelSequenceDataset(Dataset):
    """Trả về TOÀN BỘ chuỗi segment của 1 video (khoá theo (dataset, path)), sắp theo thời
    gian bắt đầu — dùng cho conditioning (cần biết thứ tự thời gian giữa các segment để lan
    truyền thông tin từ hành động chuyển tiếp sang trạng thái tĩnh theo sau)."""

    def __init__(
        self,
        sources: list[tuple[str, str]],
        feature_cache_dir: str,
        fps: float,
        stride_seconds: float,
        keep_datasets: set[str] | None = None,
    ):
        self.fps = fps
        self.stride_frames = stride_frames_from_seconds(fps, stride_seconds)
        self.feature_cache_dir = feature_cache_dir
        self._features = FeatureCache(feature_cache_dir, max_cached=4)

        rows = load_sources(sources, keep_datasets)
        videos: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            key = (row["dataset"], row["path"])
            videos.setdefault(key, []).append(row)
        for key in videos:
            videos[key].sort(key=lambda r: r["start"])

        self.video_keys = list(videos.keys())
        self.videos = videos

    def __len__(self) -> int:
        return len(self.video_keys)

    def __getitem__(self, idx: int):
        dataset, path = self.video_keys[idx]
        segs = self.videos[(dataset, path)]
        feats_full = self._features.get(dataset, path)

        feats, labels, starts = [], [], []
        for seg in segs:
            feats.append(_segment_feature_slice(
                feats_full, seg["start"], seg["end"], self.fps, self.stride_frames
            ))
            labels.append(seg["label"])
            starts.append(seg["start"])

        return {
            "features": torch.from_numpy(np.stack(feats)).float(),   # (num_segments, D)
            "labels": torch.tensor(labels, dtype=torch.long),        # (num_segments,)
            "start_times": torch.tensor(starts, dtype=torch.float),  # giây
            "dataset": dataset,
            "path": path,
        }


def sequence_collate_fn(batch: list[dict]) -> dict:
    """Collate function cho VideoLevelSequenceDataset — video có độ dài khác nhau nên cần padding."""
    max_len = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]

    padded_feats = torch.zeros(len(batch), max_len, feat_dim)
    padded_labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)  # -100 = ignore_index
    padded_starts = torch.zeros(len(batch), max_len, dtype=torch.float)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        padded_feats[i, :n] = item["features"]
        padded_labels[i, :n] = item["labels"]
        padded_starts[i, :n] = item["start_times"]
        mask[i, :n] = True

    return {
        "features": padded_feats,
        "labels": padded_labels,
        "start_times": padded_starts,
        "mask": mask,
    }
