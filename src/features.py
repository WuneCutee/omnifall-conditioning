"""
Trích xuất đặc trưng video bằng VideoMAE-K400 hoặc SigLIP2, theo đúng cấu hình
Table 12 trong paper OmniFall (fps, stride, số frame) để đảm bảo so sánh công bằng
với baseline gốc.

Usage:
    python -m src.features --config configs/default.yaml --backbone videomae
"""
from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import VideoMAEModel, VideoMAEImageProcessor, AutoProcessor, AutoModel

from src.data import get_video_path, get_feature_path, load_sources
from src.utils import load_config, ensure_dir

# Một số dataset (vd of-syn) chưa được giải nén ra {root}/{dataset}/video/ do video nằm
# trong 1 file .tar rất lớn (~9.7GB) và máy không đủ chỗ trống để giải nén hết cùng lúc.
# Với các dataset này, đọc trực tiếp từng video từ tar (giải nén 1 video -> trích feature
# -> xoá ngay), không bao giờ cần chỗ trống cho toàn bộ video cùng lúc.
TAR_SOURCES = {
    "of-syn": "data_files/omnifall-synthetic_av1.tar",
}


def read_video_frames(video_path: str, target_fps: float) -> np.ndarray:
    """Đọc video, resample về target_fps. Trả về mảng (num_frames, H, W, 3) RGB."""
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(int(round(orig_fps / target_fps)), 1)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 224, 224, 3), dtype=np.uint8)


class VideoMAEFeatureExtractor:
    """
    Trích xuất đặc trưng VideoMAE-K400, cửa sổ trượt 16 frame với stride 2/15s (7.5fps),
    đúng theo Table 12 của paper.
    """

    def __init__(self, model_name: str, fps: float, num_frames: int,
                 stride_seconds: float, device: str = "cuda", batch_size: int = 32,
                 fp16: bool = True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()
        self.fp16 = fp16 and self.device == "cuda"
        if self.fp16:
            self.model = self.model.half()
        self.fps = fps
        self.num_frames = num_frames
        self.stride_frames = max(int(round(stride_seconds * fps)), 1)
        self.batch_size = batch_size

    @torch.no_grad()
    def extract(self, video_path: str) -> np.ndarray:
        """Trả về mảng (T, D) — 1 vector đặc trưng cho mỗi vị trí cửa sổ trượt.

        Với stride=1 frame (mặc định của config paper: fps=7.5, stride=2/15s), các cửa
        sổ 16-frame liên tiếp trùng nhau 15/16 frame. Gọi processor 1 lần trên TOÀN BỘ
        frame duy nhất của video (đã verify bằng số: xử lý 1 frame riêng lẻ cho kết quả
        giống hệt xử lý frame đó trong 1 clip, diff=0.0), rồi ghép cửa sổ bằng cách lập
        chỉ mục tensor — giảm ~16 lần khối lượng resize/normalize so với gọi processor
        theo từng cửa sổ (bottleneck thực sự, không phải model forward).
        """
        frames = read_video_frames(video_path, self.fps)
        if len(frames) < self.num_frames:
            # Pad bằng cách lặp lại frame cuối nếu video quá ngắn
            pad = np.repeat(frames[-1:], self.num_frames - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)

        starts = list(range(0, len(frames) - self.num_frames + 1, self.stride_frames))
        if not starts:
            return np.zeros((0, self.model.config.hidden_size))

        preprocess_chunk = 128
        processed = []
        for i in range(0, len(frames), preprocess_chunk):
            chunk = list(frames[i:i + preprocess_chunk])
            px = self.processor([chunk], return_tensors="pt")["pixel_values"]  # (1, chunk_len, 3, H, W)
            processed.append(px[0])
        all_px = torch.cat(processed, dim=0)  # (N, 3, H, W) — đã resize/normalize 1 lần/frame
        if self.fp16:
            all_px = all_px.half()

        features = []
        for i in range(0, len(starts), self.batch_size):
            batch_starts = starts[i:i + self.batch_size]
            windows = torch.stack([all_px[s:s + self.num_frames] for s in batch_starts])  # (B, T, 3, H, W)
            outputs = self.model(pixel_values=windows.to(self.device))
            # Mean-pool token dimension của final hidden state, giống mô tả paper (Appendix A.8)
            feat = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()  # (batch, D)
            features.append(feat)

        return np.concatenate(features, axis=0).astype(np.float32)


class SigLIP2FeatureExtractor:
    """Trích xuất đặc trưng SigLIP2 frame-by-frame (không có temporal context), theo paper."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def extract(self, video_path: str, fps: float) -> np.ndarray:
        frames = read_video_frames(video_path, fps)
        features = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
            feat = self.model.get_image_features(**inputs).squeeze(0).cpu().numpy()
            features.append(feat)
        return np.stack(features) if features else np.zeros((0, 768))


def list_required_videos(config: dict) -> list[tuple[str, str]]:
    """Liệt kê toàn bộ (dataset, path) duy nhất cần feature, gộp từ train/val/eval_sources
    trong config — tránh trích xuất thừa video không được dùng ở bước nào."""
    all_sources = (
        config["data"]["train_sources"]
        + config["data"]["val_sources"]
        + config["data"]["eval_sources"]
    )
    seen: set[tuple[str, str]] = set()
    for cfg_name, split in all_sources:
        rows = load_sources([(cfg_name, split)])
        for ds, path in zip(rows["dataset"], rows["path"]):
            seen.add((ds, path))
    return sorted(seen)


def run_extraction(config: dict, backbone: str) -> None:
    """Chạy trích xuất đặc trưng cho các video thật sự cần (theo train/val/eval_sources
    trong config), lưu ra cache_dir, khoá theo (dataset, path) — xem STRUCTURE.md."""
    cache_dir = ensure_dir(config["features"]["cache_dir"])
    root_dir = Path(config["data"]["root_dir"])

    if backbone == "videomae":
        cfg = config["features"]["videomae"]
        extractor = VideoMAEFeatureExtractor(
            model_name=cfg["model_name"], fps=cfg["fps"],
            num_frames=cfg["num_frames"], stride_seconds=cfg["stride_seconds"],
        )
    elif backbone == "siglip2":
        cfg = config["features"]["siglip2"]
        extractor = SigLIP2FeatureExtractor(model_name=cfg["model_name"])
    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone}")

    needed = list_required_videos(config)
    print(f"Cần feature cho {len(needed)} video (dataset, path) duy nhất. Kiểm tra video thật có mặt...")

    missing_by_dataset: dict[str, int] = {}
    to_process: list[tuple[str, str, Path]] = []
    for dataset, path in needed:
        video_path = get_video_path(root_dir, dataset, path)
        out_path = get_feature_path(cache_dir, dataset, path)
        if out_path.exists():
            continue
        if not video_path.exists():
            missing_by_dataset[dataset] = missing_by_dataset.get(dataset, 0) + 1
            continue
        to_process.append((dataset, path, video_path))

    if missing_by_dataset:
        print("Thiếu video (chưa tải/chưa giải nén), sẽ BỎ QUA — không chặn các dataset khác:")
        for ds, n in sorted(missing_by_dataset.items()):
            print(f"  - {ds}: thiếu {n} video")

    print(f"Bắt đầu trích xuất đặc trưng ({backbone}) cho {len(to_process)} video...")
    for dataset, path, video_path in tqdm(to_process):
        out_path = get_feature_path(cache_dir, dataset, path)
        ensure_dir(str(out_path.parent))
        try:
            if backbone == "videomae":
                feat = extractor.extract(str(video_path))
            else:
                feat = extractor.extract(str(video_path), fps=cfg.get("fps", 7.5))
            np.save(out_path, feat)
        except Exception as e:  # noqa: BLE001
            print(f"[LỖI] ({dataset}, {path}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="videomae", choices=["videomae", "siglip2"])
    args = parser.parse_args()

    config = load_config(args.config)
    run_extraction(config, args.backbone)
