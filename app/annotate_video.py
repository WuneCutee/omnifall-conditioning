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


def build_engine(mode: str | None = None, model: str = "ft", alarm_on: float = 1.0) -> FallEngine:
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


def annotate(engine: FallEngine, video: str, out: str, height: int = 720, progress=None) -> dict:
    """Xử lý toàn bộ video, ghi video có nhãn ra `out`. Trả về dòng thời gian dự đoán và các lần cảnh báo."""
    engine.reset()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer, rows, i, start_t = None, [], 0, None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / fps
        if engine.push_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t):
            while engine.pending:
                engine.step()
            st = engine.state
            if st.ready:
                rows.append({"t": round(t, 2), "status": engine.status, "label": st.label,
                             "probs": [round(float(p), 3) for p in st.probs]})
        st = engine.state
        i += 1
        if progress and i % 15 == 0:
            progress(min(i / n, 1.0))
        if not st.ready:  # mô hình chưa đủ khung hình để dự đoán: không ghi phần khởi động ra video
            continue
        if start_t is None:
            start_t = t
        snap = (engine.status, st.probs, st.baseline_probs, st.t, list(st.timeline), st.latency_ms,
                list(st.alarm_events), 0)
        scale = height / bgr.shape[0]
        frame = cv2.resize(bgr, (int(bgr.shape[1] * scale), height))
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), COLORS[engine.status], 10)
        cv2.putText(frame, f"{t:5.1f}s  {st.label}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        canvas = np.hstack([frame, draw_panel(height, snap, engine.cfg.mode, fps, t, int(t * 3) % 2 == 0)])
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas.shape[1::-1])
        writer.write(canvas)
    cap.release()
    if writer is not None:
        writer.release()
    # start: thời điểm (giây, theo video gốc) của frame đầu tiên trong video kết quả
    return {"fps": fps, "duration": i / fps, "start": round(start_t or 0.0, 3), "timeline": rows,
            "alarms": [round(x, 2) for x in engine.state.alarm_events]}


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
    res = annotate(engine, args.video, args.out, args.height,
                   progress=lambda p: print(f"{p * 100:5.1f}%") if int(p * 100) % 10 == 0 else None)
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["t", "status", "label"] + CLASS_NAMES)
        w.writerows([[r["t"], r["status"], r["label"]] + r["probs"] for r in res["timeline"]])
    print(f"Xong: {args.out}\nCảnh báo tại (s): {res['alarms']}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
