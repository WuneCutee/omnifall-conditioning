"""Bước 1b — Kiểm toán kỹ thuật 818 video OF-ItW (OOPS) và tác động của tiền xử lý:

  * fps, độ phân giải, tỉ lệ khung, thời lượng, bitrate -> điểm chất lượng (cùng công thức
    với scripts/video_quality_score.py)
  * lệch thời gian của cách lấy mẫu legacy (bước nguyên round(fps/7.5)) theo từng video
  * phần diện tích khung hình bị bỏ do resize-cạnh-ngắn + center-crop 224x224
  * phân bố độ dài segment so với khung thời gian của 1 input classifier (18 feature = 4.4 s)

Xuất outputs_v2/data_audit/video_audit.csv + fig_video_*.png

Usage:
    python pipeline/s01b_video_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcfall.config import FEATURE_STRIDE, NUM_FEATURES, OOPS_VIDEO, OUTPUTS, TARGET_FPS, WINDOW_FRAMES
from tcfall.data import clean_labels, load_labels, load_split
from tcfall.video import probe, sample_indices

OUT = OUTPUTS / "data_audit"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9})


def robust_z(x):
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / mad if mad > 1e-9 else np.zeros_like(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, _ = clean_labels(load_labels("OOPS"))
    sp = load_split("OOPS")
    rows = []
    for p in sorted(labels.path.unique()):
        f = OOPS_VIDEO / f"{p}.mp4"
        info = probe(str(f))
        n, fps = info["frames"], info["fps"]
        leg = sample_indices(n, fps, "legacy")
        # thời gian thật của frame thứ j trong chuỗi legacy so với thời gian giả định j/7.5
        drift = leg / fps - np.arange(len(leg)) / TARGET_FPS
        w, h = info["width"], info["height"]
        kept = min(w, h) ** 2 / (w * h)  # resize cạnh ngắn rồi crop vuông -> giữ tỉ lệ này
        rows.append({"path": p, "split": sp.get(p), **info, "aspect": w / h,
                     "size_mb": f.stat().st_size / 1e6,
                     "bitrate_kbps": f.stat().st_size * 8 / max(info["duration"], 1e-6) / 1000,
                     "bpp": f.stat().st_size * 8 / max(w * h * n, 1),
                     "legacy_eff_fps": fps / max(int(round(fps / TARGET_FPS)), 1),
                     "legacy_drift_end_s": float(-drift[-1]) if len(drift) else 0.0,
                     "area_kept_by_center_crop_%": 100 * kept})
    df = pd.DataFrame(rows)
    q = 0.60 * robust_z(np.log(df.bpp.clip(lower=1e-6))) + 0.25 * robust_z(np.log(df.width * df.height)) \
        + 0.15 * robust_z(np.minimum(df.fps, 30))
    df["quality_score"] = 100 / (1 + np.exp(-q))
    df.to_csv(OUT / "video_audit.csv", index=False)

    # Ảnh hưởng lệch thời gian lên nhãn: tỉ lệ segment mà >50% thời lượng bị trượt khỏi cửa sổ đúng
    seg = labels.merge(df[["path", "fps", "legacy_eff_fps"]], on="path")
    seg["drift_at_mid_s"] = ((seg.start + seg.end) / 2) * (1 - TARGET_FPS / seg.legacy_eff_fps)
    seg["shift_ratio"] = seg.drift_at_mid_s.abs() / seg.dur
    seq_span = (NUM_FEATURES - 1) * FEATURE_STRIDE + WINDOW_FRAMES / TARGET_FPS

    summary = {
        "videos": len(df),
        "fps_counts": df.fps.round(2).value_counts().to_dict(),
        "portrait_videos(aspect<1)": int((df.aspect < 1).sum()),
        "area_kept_median_%": float(df["area_kept_by_center_crop_%"].median()),
        "videos_drift_>0.5s": int((df.legacy_drift_end_s.abs() > 0.5).sum()),
        "max_drift_s": float(df.legacy_drift_end_s.abs().max()),
        "segments_shift>50%_of_duration": int((seg.shift_ratio > 0.5).sum()),
        "test_segments_shift>50%": int(((seg.shift_ratio > 0.5) & (seg.path.map(sp) == "test")).sum()),
        "classifier_input_span_s": seq_span,
        "segments_shorter_than_input_span_%": float(100 * (labels.dur < seq_span).mean()),
        "segments_shorter_than_1_window_%": float(100 * (labels.dur < WINDOW_FRAMES / TARGET_FPS).mean()),
    }
    pd.Series(summary).to_json(OUT / "video_audit_summary.json", indent=2, force_ascii=False)
    for k, v in summary.items():
        print(f"{k}: {v}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    for fps, g in df.groupby(df.fps.round(2)):
        ax.scatter(g.duration, g.legacy_drift_end_s, s=8, label=f"{fps} fps (n={len(g)})")
    ax.set_xlabel("Thời lượng video (s)"); ax.set_ylabel("Lệch thời gian cuối video (s)")
    ax.set_title("Lấy mẫu legacy: lệch nhãn-feature"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[1]
    ax.hist(df["area_kept_by_center_crop_%"], bins=30, color="#4a90d9")
    ax.set_xlabel("% diện tích khung giữ lại sau center-crop"); ax.set_ylabel("số video")
    ax.set_title("Mất thông tin do center-crop"); ax.grid(alpha=.3)
    ax = axes[2]
    ax.hist(labels.dur.clip(upper=12), bins=48, color="#e07b39")
    ax.axvline(WINDOW_FRAMES / TARGET_FPS, color="k", ls="--", lw=1, label="1 cửa sổ VideoMAE (2.13 s)")
    ax.axvline(seq_span, color="r", ls="--", lw=1, label=f"1 input classifier ({seq_span:.2f} s)")
    ax.set_xlabel("Độ dài segment ItW (s, cắt ở 12)"); ax.set_ylabel("số segment")
    ax.set_title("Segment ngắn hơn ngữ cảnh mô hình"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_video_audit.png"); plt.close(fig)


if __name__ == "__main__":
    main()
