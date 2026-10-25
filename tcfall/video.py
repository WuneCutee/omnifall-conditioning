"""Đọc video và lấy mẫu khung hình về 7.5 fps.

Hai cách lấy mẫu:
  * legacy  — cách cũ trong src/features.py: lấy 1 frame mỗi round(fps/7.5) frame gốc.
              Với video 23.98 fps, bước = 3 -> thực tế 7.99 fps, nhưng feature vẫn được gán
              mốc thời gian i * 2/15 s -> lệch dần so với nhãn (tới ~2 s ở cuối video dài).
  * timefix — chọn frame gốc gần nhất với mốc t_j = j / 7.5 s, đúng 7.5 fps với mọi fps gốc.
Cả hai được tạo từ CÙNG một lượt giải mã video (đọc tuần tự, chỉ giữ frame cần dùng).
"""
from __future__ import annotations

import cv2
import numpy as np

from tcfall.config import TARGET_FPS


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    info["duration"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0.0
    return info


def sample_indices(n_frames: int, fps: float, variant: str, target_fps: float = TARGET_FPS) -> np.ndarray:
    """Chỉ số frame gốc được giữ lại, theo thứ tự thời gian."""
    if fps <= 0 or n_frames <= 0:
        return np.zeros(0, dtype=int)
    if variant == "legacy":
        step = max(int(round(fps / target_fps)), 1)
        return np.arange(0, n_frames, step)
    if variant == "timefix":
        duration = n_frames / fps
        times = np.arange(0.0, duration, 1.0 / target_fps)
        idx = np.round(times * fps).astype(int)
        return np.clip(idx, 0, n_frames - 1)
    raise ValueError(variant)


def read_sampled(path: str, variants: tuple[str, ...]) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Giải mã video 1 lần, trả về {variant: list frame RGB uint8} và thông tin video.
    Frame dùng chung giữa các biến thể chỉ được lưu 1 lần (list chứa tham chiếu)."""
    info = probe(path)
    wanted = {v: sample_indices(info["frames"], info["fps"], v) for v in variants}
    needed = set()
    for idx in wanted.values():
        needed.update(idx.tolist())

    cap = cv2.VideoCapture(path)
    frames: dict[int, np.ndarray] = {}
    last = max(needed) if needed else -1
    i = 0
    while i <= last:
        if i in needed:
            ok, fr = cap.read()
            if not ok:
                break
            frames[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        else:
            if not cap.grab():
                break
        i += 1
    cap.release()

    out = {}
    decoded_max = max(frames) if frames else -1
    for v, idx in wanted.items():
        # Số frame thật giải mã được có thể ít hơn CAP_PROP_FRAME_COUNT -> dùng frame cuối hợp lệ
        idx = np.minimum(idx, decoded_max) if decoded_max >= 0 else idx[:0]
        out[v] = [frames[j] for j in idx]
    info["decoded_frames"] = decoded_max + 1
    return out, info
