"""Tiền xử lý frame trên GPU cho chế độ realtime, tương đương VideoMAEImageProcessor:
resize cạnh ngắn -> 224 (bilinear, antialias như PIL), center-crop 224x224, /255,
chuẩn hoá mean/std ImageNet. Nhanh hơn processor CPU nhiều lần; độ tương đương được
đo trong pipeline/s06_check_gpu_preprocess.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_frames(frames_rgb: list[np.ndarray] | np.ndarray, device: str, fp16: bool = True,
                      size: int = 224, crops: tuple[str, ...] = ("center",)) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB -> (N, 3, 224, 224) đã chuẩn hoá, trên device.
    Nhiều crop ("center", "start", "end") -> (len(crops), N, 3, 224, 224)."""
    x = torch.from_numpy(np.ascontiguousarray(np.stack(frames_rgb) if isinstance(frames_rgb, list) else frames_rgb))
    x = x.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    if h <= w:
        nh, nw = size, int(size * w / h)
    else:
        nh, nw = int(size * h / w), size
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    x = (x / 255.0 - MEAN.to(device)) / STD.to(device)
    outs = []
    for c in crops:
        top = {"center": (nh - size) // 2, "start": 0, "end": nh - size}[c]
        left = {"center": (nw - size) // 2, "start": 0, "end": nw - size}[c]
        outs.append(x[..., top:top + size, left:left + size])
    out = outs[0] if len(crops) == 1 else torch.stack(outs)
    return out.half() if fp16 else out
