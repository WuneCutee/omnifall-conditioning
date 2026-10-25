"""Ứng dụng giám sát té ngã realtime (webcam hoặc video), giao diện OpenCV.

Kiến trúc 3 luồng: [camera] -> hàng đợi -> [suy luận GPU] -> trạng thái -> [giao diện]
nên hình ảnh luôn mượt kể cả khi suy luận chậm hơn tốc độ camera.

Chạy:
    python app/realtime_app.py                       # webcam 0, chế độ rule
    python app/realtime_app.py --mode hybrid
    python app/realtime_app.py --video demo_videos/xxx.mp4      # phát theo tốc độ thật
    python app/realtime_app.py --video xxx.mp4 --record out.mp4

Phím: 1 baseline | 2 rule | 3 hybrid | 4 hmm | r reset | s chụp màn hình | space tạm dừng | q thoát
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from tcfall.config import CLASS_NAMES, FALL, FALLEN, OUTPUTS
from tcfall.realtime import MODES, EngineConfig, FallEngine

PANEL_W = 380
COLORS = {"NORMAL": (60, 180, 60), "FALL": (0, 190, 240), "FALLEN": (0, 80, 255), "ALARM": (0, 0, 230),
          "WARMUP": (160, 160, 160)}
CLASS_COLORS = [(180, 180, 180), (0, 190, 240), (0, 0, 230), (200, 160, 60), (200, 120, 40),
                (180, 80, 200), (140, 40, 160), (120, 200, 120), (60, 150, 60), (110, 110, 110)]


def load_best_config() -> dict:
    f = OUTPUTS / "best_config.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


class Capture(threading.Thread):
    """Đọc frame liên tục. Webcam: mốc thời gian theo đồng hồ; video: theo chỉ số frame,
    phát đúng tốc độ gốc (realtime) để mô phỏng camera."""

    def __init__(self, source, is_file: bool, out_q: queue.Queue):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if (not is_file and sys.platform == "win32") else cv2.CAP_ANY)
        if not is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.is_file, self.q = is_file, out_q
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.latest, self.latest_t = None, 0.0
        self.lock = threading.Lock()
        self.running, self.paused, self.ended = True, False, False
        self.measured_fps = 0.0

    def run(self):
        t0, n, tick = time.perf_counter(), 0, time.perf_counter()
        while self.running:
            if self.paused:
                time.sleep(0.05); t0 += 0.05; continue
            ok, frame = self.cap.read()
            if not ok:
                self.ended = True
                break
            n += 1
            if self.is_file:
                t = n / self.fps
                delay = t0 + t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            else:
                t = time.perf_counter() - t0
            with self.lock:
                self.latest, self.latest_t = frame, t
            try:
                self.q.put_nowait((frame, t))
            except queue.Full:
                pass  # suy luận quá tải: bỏ frame (engine vẫn lấy mẫu theo thời gian)
            if n % 30 == 0:
                now = time.perf_counter(); self.measured_fps = 30 / (now - tick); tick = now
        self.cap.release()


class Inference(threading.Thread):
    def __init__(self, engine: FallEngine, in_q: queue.Queue):
        super().__init__(daemon=True)
        self.engine, self.q = engine, in_q
        self.lock = threading.Lock()
        self.running = True
        self.reset_flag = False
        self.snapshot = None

    def run(self):
        import cv2 as _cv2
        while self.running:
            if self.reset_flag:
                self.engine.reset(); self.reset_flag = False
            got = False
            try:
                while True:
                    frame, t = self.q.get(timeout=0.01 if not got else 0)
                    got = True
                    self.engine.push_frame(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB), t)
            except queue.Empty:
                pass
            st = self.engine.step()
            with self.lock:
                self.snapshot = (self.engine.status, st.probs.copy(), st.baseline_probs.copy(), st.t,
                                 list(st.timeline), st.latency_ms, list(st.alarm_events), len(self.engine.pending))


def draw_panel(h: int, snap, mode: str, cap_fps: float, video_t: float, blink: bool) -> np.ndarray:
    panel = np.full((h, PANEL_W, 3), 28, np.uint8)
    put = lambda txt, y, s=0.55, c=(230, 230, 230), th=1: cv2.putText(panel, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)
    put("OmniFall Transition-Conditioning", 28, 0.55, (255, 255, 255), 1)
    put(f"mode: {mode}   [1-4 doi che do]", 52, 0.45, (170, 170, 170))
    if snap is None:
        put("Dang tai mo hinh...", 100, 0.7); return panel
    status, probs, bprobs, t, timeline, lat, events, pending = snap
    col = COLORS[status]
    if status == "ALARM" and not blink:
        col = (0, 0, 120)
    cv2.rectangle(panel, (10, 66), (PANEL_W - 10, 120), col, -1)
    label = {"ALARM": "CANH BAO: NGA - CAN HO TRO", "FALLEN": "FALLEN (dang theo doi)", "FALL": "FALL",
             "NORMAL": "BINH THUONG", "WARMUP": "Khoi dong (~5 s)"}[status]
    cv2.putText(panel, label, (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    y = 148
    put("Xac suat (thanh mau: sau dieu kien hoa, vach xam: baseline)", y, 0.36, (170, 170, 170))
    y += 10
    for k, name in enumerate(CLASS_NAMES):
        y += 22
        put(name, y, 0.45)
        x0, wmax = 110, PANEL_W - 130
        cv2.rectangle(panel, (x0, y - 12), (x0 + int(wmax * probs[k]), y + 2), CLASS_COLORS[k], -1)
        bx = x0 + int(wmax * bprobs[k])
        cv2.line(panel, (bx, y - 14), (bx, y + 4), (200, 200, 200), 2)
        cv2.putText(panel, f"{probs[k]*100:3.0f}%", (PANEL_W - 52, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # dòng thời gian 30 s gần nhất
    y += 34
    put("30 s gan nhat", y, 0.42, (170, 170, 170))
    y0, x0, x1 = y + 8, 14, PANEL_W - 14
    cv2.rectangle(panel, (x0, y0), (x1, y0 + 22), (60, 60, 60), -1)
    for tt, k in timeline:
        if t - tt <= 30:
            x = int(x1 - (t - tt) / 30 * (x1 - x0))
            cv2.line(panel, (x, y0), (x, y0 + 22), CLASS_COLORS[k], 2)
    y = y0 + 48
    lag = max(0.0, video_t - t)
    put(f"camera: {cap_fps:4.1f} fps   tre: {lag:4.2f} s", y, 0.45); y += 22
    put(f"suy luan: {lat:5.1f} ms/cua so   hang doi: {pending}", y, 0.45); y += 22
    put(f"so lan canh bao: {len(events)}", y, 0.45); y += 30
    put("r reset | s chup | space dung | q thoat", h - 16, 0.42, (150, 150, 150))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default=None)
    ap.add_argument("--record", default=None, help="Ghi màn hình ứng dụng ra mp4")
    ap.add_argument("--height", type=int, default=540, help="Chiều cao khung video hiển thị")
    ap.add_argument("--beep", action="store_true", help="Phát âm báo khi có cảnh báo (Windows)")
    ap.add_argument("--spatial", choices=["auto", "on", "off"], default="auto",
                    help="TTA không gian 3 crop (x3 GPU). auto = theo cấu hình đã chọn trên val")
    ap.add_argument("--model", choices=["ft", "base"], default="ft",
                    help="ft = classifier fine-tune ItW-train (mặc định), base = checkpoint St+Sy gốc")
    ap.add_argument("--baseline_ckpt", default=None,
                    help="Classifier đã fine-tune trên ItW-train (outputs_v2/finetune/baseline_ft_itw_seed42.pt)")
    ap.add_argument("--alarm_on", type=float, default=1.0,
                    help="Số giây 'fallen' liên tục trước khi báo động (thấp: nhạy hơn, nhiều báo giả hơn)")
    args = ap.parse_args()

    # Cấu hình streaming đã tinh chỉnh trên ItW-val (pipeline/s07_stream_eval.py).
    # Mặc định dùng classifier đã fine-tune trên ItW-train nếu có (tốt nhất khi streaming).
    best = load_best_config()
    ft_default = OUTPUTS / "finetune" / "baseline_ft_itw_seed42.pt"
    if args.model == "ft" and not args.baseline_ckpt and ft_default.exists():
        args.baseline_ckpt = str(ft_default)
    st = best.get("stream_ft" if args.baseline_ckpt else "stream", best.get("stream", {}))
    cfg = EngineConfig(mode=args.mode or st.get("mode", "rule"),
                       tau=st.get("tau", 4.0), window=st.get("window", 10.0),
                       segment_hop=st.get("segment_hop", 1.0), alarm_on_seconds=args.alarm_on,
                       spatial_tta=(args.spatial == "on") or (args.spatial == "auto" and st.get("input") == "spatial"))
    if st.get("hmm_A") is not None:
        cfg.hmm_A = np.array(st["hmm_A"]); cfg.hmm_prior = np.array(st["hmm_prior"])
        cfg.hmm_strength = st.get("hmm_strength", 1.0)
    print(f"Cấu hình: mode={cfg.mode} hop={cfg.segment_hop}s spatial_tta={cfg.spatial_tta} "
          f"alarm_on={cfg.alarm_on_seconds}s ({st.get('name', 'mặc định')})")

    if args.baseline_ckpt:
        import os
        os.environ["TCFALL_BASELINE_CKPT"] = args.baseline_ckpt
    print("Đang tải VideoMAE + baseline + conditioner ...")
    engine = FallEngine(cfg)
    q: queue.Queue = queue.Queue(maxsize=64)
    is_file = args.video is not None
    cap = Capture(args.video if is_file else args.camera, is_file, q)
    if not cap.cap.isOpened():
        print(f"Không mở được nguồn: {args.video or args.camera}"); return
    inf = Inference(engine, q)
    cap.start(); inf.start()

    log_dir = OUTPUTS / "app_logs"; log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / "alarm_events.csv", "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    writer, n_events = None, 0
    win = "OmniFall realtime"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            with cap.lock:
                frame, vt = (None, 0.0) if cap.latest is None else (cap.latest.copy(), cap.latest_t)
            with inf.lock:
                snap = inf.snapshot
            if frame is None:
                frame = np.zeros((args.height, int(args.height * 16 / 9), 3), np.uint8)
            scale = args.height / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), args.height))
            blink = int(time.time() * 3) % 2 == 0
            if snap is not None:
                status = snap[0]
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), COLORS[status], 8)
                if len(snap[6]) > n_events:
                    for t_ev in snap[6][n_events:]:
                        log.writerow([datetime.now().isoformat(timespec="seconds"), args.video or f"camera{args.camera}",
                                      engine.cfg.mode, f"{t_ev:.2f}"]); log_f.flush()
                        print(f"[CẢNH BÁO] phát hiện FALLEN kéo dài tại t={t_ev:.1f}s")
                        if args.beep and sys.platform == "win32":
                            import winsound
                            winsound.Beep(1200, 300)
                    n_events = len(snap[6])
            canvas = np.hstack([frame, draw_panel(args.height, snap, engine.cfg.mode, cap.measured_fps, vt, blink)])
            cv2.imshow(win, canvas)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, canvas.shape[1::-1])
                writer.write(canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                engine.cfg.mode = MODES[key - ord("1")]
            elif key == ord("r"):
                inf.reset_flag = True; n_events = 0
            elif key == ord("s"):
                p = log_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S}.png"; cv2.imwrite(str(p), canvas); print("Đã lưu", p)
            elif key == ord(" "):
                cap.paused = not cap.paused
            if cap.ended and q.empty() and not engine.pending:
                time.sleep(1.0); break
    finally:
        cap.running = inf.running = False
        log_f.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
