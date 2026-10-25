"""Bước 6 — Kiểm tra tiền xử lý GPU (dùng trong app) có tương đương VideoMAEImageProcessor
(dùng khi trích feature offline) không: so sánh pixel, feature VideoMAE và dự đoán baseline
trên các video OOPS-test; đo tốc độ của 2 cách.

Usage:
    python pipeline/s06_check_gpu_preprocess.py [--videos 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from tcfall.baseline import load_baseline
from tcfall.config import NUM_FEATURES, OOPS_VIDEO, OUTPUTS
from tcfall.data import load_split
from tcfall.features import VideoMAEExtractor
from tcfall.preprocess_gpu import preprocess_frames
from tcfall.video import read_sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=20)
    args = ap.parse_args()
    ext = VideoMAEExtractor()
    base = load_baseline(ext.device)
    paths = sorted(p for p, s in load_split("OOPS").items() if s == "test")[:: max(1, 572 // args.videos)][: args.videos]
    px_err, cos, agree, t_cpu, t_gpu, n_fr = [], [], [], 0.0, 0.0, 0
    for p in paths:
        frames = read_sampled(str(OOPS_VIDEO / f"{p}.mp4"), ("timefix",))[0]["timefix"]
        t0 = time.perf_counter(); a = ext.preprocess(frames).float(); t_cpu += time.perf_counter() - t0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        b = preprocess_frames(frames, ext.device, fp16=False); torch.cuda.synchronize(); t_gpu += time.perf_counter() - t0
        n_fr += len(frames)
        px_err.append(float((a.to(b.device) - b).abs().mean()))
        fa = ext.embed_windows(a.half() if ext.fp16 else a)
        fb = ext.embed_windows(b.half() if ext.fp16 else b)
        cos.append(float(((fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1))).mean()))
        if len(fa) >= NUM_FEATURES:
            sa = torch.from_numpy(np.stack([fa[i:i + NUM_FEATURES] for i in range(len(fa) - NUM_FEATURES + 1)]))
            sb = torch.from_numpy(np.stack([fb[i:i + NUM_FEATURES] for i in range(len(fb) - NUM_FEATURES + 1)]))
            with torch.no_grad():
                la = base(sa.to(ext.device))["logits"].argmax(-1)
                lb = base(sb.to(ext.device))["logits"].argmax(-1)
            agree.append(float((la == lb).float().mean()))
    res = {"videos": len(paths), "frames": n_fr,
           "pixel_mean_abs_diff(normalized units)": float(np.mean(px_err)),
           "feature_cosine_mean": float(np.mean(cos)),
           "baseline_argmax_agreement_%": 100 * float(np.mean(agree)),
           "cpu_processor_ms_per_frame": 1000 * t_cpu / n_fr, "gpu_ms_per_frame": 1000 * t_gpu / n_fr}
    (OUTPUTS / "gpu_preprocess_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
