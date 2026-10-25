"""
Chấm điểm chất lượng video từ tỉ lệ dung lượng file / thời lượng file.

Với mỗi video:
  - bitrate (kbps)        = size * 8 / duration / 1000
  - bits-per-pixel (bpp)  = size * 8 / (width * height * số khung)   (chuẩn hoá theo độ phân giải)
Điểm chất lượng (0-100) = sigmoid của tổ hợp có trọng số các z-score "chắc chắn" (median/MAD):
    q = w_bpp * z(log bpp) + w_res * z(log số pixel) + w_fps * z(min(fps, 30))
    score = 100 / (1 + exp(-q))
Video bpp thấp bị nén mạnh (mờ, vỡ khối) nên điểm thấp. Chia 3 mức: LOW / MID / HIGH theo --low, --high.

Usage:
    python -m scripts.video_quality_score --video_dir data/omnifall/OOPS/video/falls
    python -m scripts.video_quality_score --video_dir ... --low 25 --out quality.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

W_BPP, W_RES, W_FPS = 0.60, 0.25, 0.15
EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def probe(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if fps <= 0 or frames <= 0 or w <= 0 or h <= 0:
        return None
    size = path.stat().st_size
    duration = frames / fps
    return {
        "video": path.stem, "size_mb": size / 1e6, "duration_s": duration, "fps": fps,
        "width": int(w), "height": int(h), "frames": int(frames),
        "bitrate_kbps": size * 8 / duration / 1000,
        "bpp": size * 8 / (w * h * frames),
    }


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-9:
        return np.zeros_like(x)
    return (x - med) / mad


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    z_bpp = robust_z(np.log(df["bpp"].clip(lower=1e-6).to_numpy()))
    z_res = robust_z(np.log((df["width"] * df["height"]).to_numpy(dtype=float)))
    z_fps = robust_z(np.minimum(df["fps"].to_numpy(), 30.0))
    q = W_BPP * z_bpp + W_RES * z_res + W_FPS * z_fps
    df = df.copy()
    df["z_bpp"], df["z_res"], df["z_fps"] = z_bpp, z_res, z_fps
    df["score"] = 100.0 / (1.0 + np.exp(-q))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out", default="data/omnifall/video_quality.csv")
    parser.add_argument("--low", type=float, default=25.0, help="Điểm < low => LOW")
    parser.add_argument("--high", type=float, default=75.0, help="Điểm >= high => HIGH")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.video_dir).rglob("*") if p.suffix.lower() in EXTS)
    print(f"Tìm thấy {len(files)} video")
    rows, bad = [], 0
    for i, p in enumerate(files, 1):
        r = probe(p)
        if r is None:
            bad += 1
            continue
        rows.append(r)
        if i % 200 == 0:
            print(f"  đã đo {i}/{len(files)}")
    df = score_table(pd.DataFrame(rows))
    df["tier"] = np.where(df["score"] < args.low, "LOW", np.where(df["score"] >= args.high, "HIGH", "MID"))
    df.sort_values("score").to_csv(args.out, index=False)

    print(f"\nKhông đọc được: {bad} video")
    print(df[["size_mb", "duration_s", "fps", "bitrate_kbps", "bpp", "score"]].describe().round(3).to_string())
    print("\nPhân bố mức chất lượng:")
    print(df["tier"].value_counts().to_string())
    print(f"\nTương quan Spearman(score, bitrate): {df['score'].corr(df['bitrate_kbps'], method='spearman'):.3f}")
    print("\n5 video điểm thấp nhất:")
    print(df.nsmallest(5, "score")[["video", "bitrate_kbps", "bpp", "width", "height", "score"]].round(3).to_string(index=False))
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
