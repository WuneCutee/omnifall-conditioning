"""Chạy mô hình trên 1 video (offline, xử lý mọi frame, không bỏ frame) và ghi kết quả lên video:
khung màu theo trạng thái + bảng xác suất 10 lớp bên phải. Kèm file CSV dự đoán theo thời gian.

Chạy:
    python app/annotate_video.py --video input.mp4 --out output.mp4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from tcfall.config import CLASS_NAMES, OUTPUTS
from tcfall.realtime import MODES, EngineConfig, FallEngine
from app.realtime_app import COLORS, draw_panel, load_best_config


def build_engine(mode: str | None, model: str, alarm_on: float) -> FallEngine:
    best = load_best_config()
    ckpt = OUTPUTS / "finetune" / "baseline_ft_itw_seed42.pt"
    use_ft = model == "ft" and ckpt.exists()
    if use_ft:
        os.environ["TCFALL_BASELINE_CKPT"] = str(ckpt)
    st = best.get("stream_ft" if use_ft else "stream", best.get("stream", {}))
    cfg = EngineConfig(mode=mode or st.get("mode", "rule"), tau=st.get("tau", 4.0), window=st.get("window", 10.0),
                       segment_hop=st.get("segment_hop", 1.0), alarm_on_seconds=alarm_on,
                       spatial_tta=st.get("input") == "spatial")
    if st.get("hmm_A") is not None:
        cfg.hmm_A = np.array(st["hmm_A"]); cfg.hmm_prior = np.array(st["hmm_prior"])
        cfg.hmm_strength = st.get("hmm_strength", 1.0)
    print(f"model={'ft' if use_ft else 'base'} mode={cfg.mode} hop={cfg.segment_hop}s spatial_tta={cfg.spatial_tta}")
    return FallEngine(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=MODES, default=None)
    ap.add_argument("--model", choices=["ft", "base"], default="ft")
    ap.add_argument("--alarm_on", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    engine = build_engine(args.mode, args.model, args.alarm_on)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer, rows, i = None, [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / fps
        if engine.push_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t):
            engine.step()
            while engine.pending:
                engine.step()
            st = engine.state
            if st.ready:
                rows.append([f"{t:.2f}", engine.status, st.label] + [f"{p:.3f}" for p in st.probs])
        st = engine.state
        snap = (engine.status, st.probs, st.baseline_probs, st.t, list(st.timeline), st.latency_ms,
                list(st.alarm_events), 0)
        scale = args.height / bgr.shape[0]
        frame = cv2.resize(bgr, (int(bgr.shape[1] * scale), args.height))
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), COLORS[engine.status], 10)
        cv2.putText(frame, f"{t:5.1f}s  {st.label if st.ready else 'warmup'}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        canvas = np.hstack([frame, draw_panel(args.height, snap, engine.cfg.mode, fps, t, int(t * 3) % 2 == 0)])
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas.shape[1::-1])
        writer.write(canvas)
        i += 1
        if i % 300 == 0:
            print(f"{i}/{n} frame")
    cap.release(); writer.release()
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["t", "status", "label"] + CLASS_NAMES); w.writerows(rows)
    print(f"Xong: {args.out}\nCảnh báo tại (s): {[round(x, 1) for x in engine.state.alarm_events]}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
