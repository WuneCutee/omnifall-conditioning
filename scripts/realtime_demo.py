"""
Demo real-time phát hiện fall/fallen qua webcam (hoặc video file để test), dùng lại
NGUYÊN VẸN pipeline đã train: VideoMAE-K400 baseline (checkpoint-7000, official repo) +
RuleBasedConditioner / Hybrid v8 (LearnableConditioner, seed42) từ src/conditioning.py.

Cấu hình trích đặc trưng giữ NGUYÊN theo Table 12 (paper) và đúng những gì baseline
được train/eval trên đó — xem src/features.py:
    fps=7.5, num_frames=16 (1 cửa sổ VideoMAE), stride=2/15s (~1 resampled-frame)
Classifier (FeatureTransformer) cần 18 đặc trưng VideoMAE liên tiếp làm 1 input
(NUM_FEATURES=18, xem cache_baseline_logits_stsyn.py) -> 1 dự đoán "hiện tại".

Usage:
    # Webcam trực tiếp (camera 0), cả 2 chế độ song song:
    python -m scripts.realtime_demo --mode both

    # Chỉ rule-based (nhẹ nhất):
    python -m scripts.realtime_demo --mode rule_based

    # Chỉ hybrid v8:
    python -m scripts.realtime_demo --mode hybrid

    # Test bằng video file có sẵn thay vì ngồi chờ webcam (xử lý nhanh hết mức có thể,
    # KHÔNG chờ theo tốc độ thật của video):
    python -m scripts.realtime_demo --mode both --video path/to/clip.mp4 --no_display

    # Không mở cửa sổ hiển thị (headless / debug), chỉ lưu overlay ra ảnh mỗi N frame:
    python -m scripts.realtime_demo --mode both --no_display --save_every 15 --save_dir demo_debug
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / "omnifall-experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from safetensors.torch import load_file  # noqa: E402
from transformers import VideoMAEImageProcessor, VideoMAEModel  # noqa: E402

from models.feature_transformer import FeatureTransformer  # noqa: E402
from src.conditioning import LearnableConditioner, RuleBasedConditioner  # noqa: E402
from src.data import CLASS_NAMES  # noqa: E402

# ============================== Cấu hình cố định (Table 12 / đã train) ==============================
VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
TARGET_FPS = 7.5
NUM_FRAMES_PER_WINDOW = 16          # 1 cửa sổ VideoMAE = 1 raw feature (768-dim)
STRIDE_SECONDS = 2 / 15             # ~1 resampled-frame giữa 2 cửa sổ liên tiếp
NUM_FEATURES = 18                   # số raw feature liên tiếp classifier cần (khớp lúc train)
FEATURE_DIM = 768
HIDDEN_DIM = 768
NUM_CLASSES = 10
NUM_LAYERS = 2
NUM_HEADS = 8

HYBRID_HIDDEN_DIM = 128
HYBRID_NUM_HEADS = 4
HYBRID_CONTEXT_WINDOW_SECONDS = 15.0
HYBRID_DROPOUT = 0.2

# Bao nhiêu giây lịch sử giữ lại cho các conditioner (>= max(boost_window rule-based=10s,
# context_window hybrid=15s), thêm biên an toàn).
HISTORY_WINDOW_SECONDS = 20.0

BASELINE_CHECKPOINT = (
    EXPERIMENTS_ROOT / "outputs" / "2026-09-08" / "21-55-48" /
    "feature_transformer_768_2l" / "checkpoint-7000" / "model.safetensors"
)
HYBRID_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "learnable_conditioner_v8_stsyn_kl0.3_seed42_best.pt"
LOG_PATH = PROJECT_ROOT / "fallen_events_log.csv"

CLASS_NAMES_10 = CLASS_NAMES[:NUM_CLASSES]
FALL_ID = CLASS_NAMES_10.index("fall")
FALLEN_ID = CLASS_NAMES_10.index("fallen")

# Màu overlay (BGR, giống OpenCV)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 200, 230)
COLOR_RED = (0, 0, 255)
COLOR_DARK_RED = (0, 0, 90)


# ============================== Model loading ==============================
def load_baseline_model(device: str) -> FeatureTransformer:
    model = FeatureTransformer(
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_features=NUM_FEATURES,
    )
    state_dict = load_file(str(BASELINE_CHECKPOINT))
    state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_hybrid_model(device: str) -> LearnableConditioner:
    model = LearnableConditioner(
        feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=HYBRID_HIDDEN_DIM,
        num_heads=HYBRID_NUM_HEADS, context_window_seconds=HYBRID_CONTEXT_WINDOW_SECONDS,
        residual=True, dropout=HYBRID_DROPOUT,
    )
    model.load_state_dict(torch.load(str(HYBRID_CHECKPOINT), map_location=device, weights_only=True))
    return model.to(device).eval()


class OnlineVideoMAEExtractor:
    """Trích 1 vector đặc trưng (768-dim) cho 1 cửa sổ 16 frame, dùng cho streaming
    (không cần toàn bộ video như VideoMAEFeatureExtractor trong src/features.py)."""

    def __init__(self, device: str):
        self.device = device
        self.processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL_NAME)
        self.model = VideoMAEModel.from_pretrained(VIDEOMAE_MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def extract(self, frames_rgb: list[np.ndarray]) -> np.ndarray:
        px = self.processor([frames_rgb], return_tensors="pt")["pixel_values"].to(self.device)
        out = self.model(pixel_values=px)
        feat = out.last_hidden_state.mean(dim=1)  # mean-pool token dim, giống src/features.py
        return feat.squeeze(0).float().cpu().numpy()


# ============================== Conditioner helpers (online, dùng lại đúng class đã train) ==============================
def rule_based_adjust(conditioner: RuleBasedConditioner, history: list[dict], device: str) -> np.ndarray:
    """Áp RuleBasedConditioner lên TOÀN BỘ cửa sổ lịch sử hiện có, trả về probs đã điều
    chỉnh cho MỌI segment trong lịch sử (batch=1) — dùng chỉ số cuối cho dự đoán "hiện tại"."""
    probs = torch.tensor(np.stack([h["baseline_probs"] for h in history]), dtype=torch.float32, device=device)
    starts = torch.tensor([h["start_time"] for h in history], dtype=torch.float32, device=device)
    mask = torch.ones(len(history), dtype=torch.bool, device=device)
    adjusted = conditioner.apply(probs.unsqueeze(0), starts.unsqueeze(0), mask.unsqueeze(0))
    return adjusted.squeeze(0).cpu().numpy()  # (seq_len, num_classes)


@torch.no_grad()
def hybrid_adjust(hybrid_model: LearnableConditioner, rule_adjusted_probs_seq: np.ndarray,
                   history: list[dict], device: str) -> np.ndarray:
    """Input của Hybrid v8 lúc train là log(rule_adjusted_probs + eps) (xem
    cache_rule_adjusted_logprobs*.py + train_learnable_conditioner_v8.py) — giữ đúng."""
    eps = 1e-8
    log_probs = np.log(rule_adjusted_probs_seq + eps)
    feats = torch.tensor(log_probs, dtype=torch.float32, device=device).unsqueeze(0)
    starts = torch.tensor([h["start_time"] for h in history], dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, len(history), dtype=torch.bool, device=device)
    logits = hybrid_model(feats, starts, mask)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs  # (seq_len, num_classes)


# ============================== Display helpers ==============================
def status_color(pred_id: int) -> tuple[int, int, int]:
    if pred_id == FALLEN_ID:
        return COLOR_RED
    if pred_id == FALL_ID:
        return COLOR_YELLOW
    return COLOR_GREEN


def draw_border(frame: np.ndarray, color: tuple[int, int, int], thickness: int = 12) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def _put_text_fit(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
                   color: tuple[int, int, int], thickness: int, max_width: int) -> None:
    """cv2.putText với viền đen + tự thu nhỏ font nếu chữ rộng hơn max_width (tránh bị
    cắt chữ trên khung hình nhỏ, vd webcam độ phân giải thấp)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > max_width:
        scale = scale * (max_width / tw)
        thickness = max(1, int(round(thickness * (max_width / tw))))
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]],
                  measured_fps: float, blink_on: bool, any_fallen: bool,
                  gt_text: str | None = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale_factor = w / 640.0  # font/border scale theo độ rộng khung hình thực tế
    margin = int(20 * scale_factor)
    max_text_width = w - 2 * margin

    worst_color = COLOR_GREEN
    if any(c == COLOR_RED for _, c in lines):
        worst_color = COLOR_RED
    elif any(c == COLOR_YELLOW for _, c in lines):
        worst_color = COLOR_YELLOW

    border_color = worst_color
    if any_fallen:
        border_color = COLOR_RED if blink_on else COLOR_DARK_RED
    draw_border(out, border_color, thickness=max(4, int(12 * scale_factor)))

    y = int(40 * scale_factor)
    line_height = int(35 * scale_factor)
    for text, color in lines:
        _put_text_fit(out, text, (margin, y), 0.8 * scale_factor, color, 2, max_text_width)
        y += line_height

    if any_fallen and blink_on:
        warn = "FALLEN - CAN KIEM TRA"
        _put_text_fit(out, warn, (margin, y + int(15 * scale_factor)), 1.0 * scale_factor,
                       COLOR_RED, 3, max_text_width)

    fps_text = f"processed: {measured_fps:.2f} segment/s"
    _put_text_fit(out, fps_text, (margin, h - int(15 * scale_factor)), 0.6 * scale_factor,
                  (255, 255, 255), 2, max_text_width)
    if gt_text:
        _put_text_fit(out, gt_text, (margin, h - int(50 * scale_factor)), 0.8 * scale_factor,
                      (255, 220, 60), 2, max_text_width)
    return out


def load_ground_truth(video_path: str):
    """Nhãn thật (start, end, tên lớp) nếu video thuộc OOPS/OF-ItW, ngược lại None."""
    import pandas as pd
    csv_path = PROJECT_ROOT / "data" / "omnifall" / "labels" / "dataset" / "OOPS.csv"
    if not csv_path.exists():
        return None
    key = "falls/" + Path(video_path).stem
    df = pd.read_csv(csv_path)
    df = df[df["path"] == key].sort_values("start")
    if df.empty:
        return None
    return [(float(r.start), float(r.end), CLASS_NAMES[int(r.label)]) for r in df.itertuples()]


def gt_at(segments, t: float) -> str | None:
    if not segments:
        return None
    for s, e, name in segments:
        if s <= t < e:
            return f"GT: {name}"
    return None


def log_fallen_event(csv_writer, mode: str, elapsed: float) -> None:
    csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), f"{elapsed:.2f}", mode, "fallen"])


# ============================== Main loop ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rule_based", "hybrid", "both"], default="both")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--video", type=str, default=None, help="Dùng video file thay vì webcam (để test).")
    parser.add_argument("--no_display", action="store_true", help="Không mở cửa sổ cv2.imshow (headless).")
    parser.add_argument("--save_every", type=int, default=0, help="Lưu overlay ra PNG mỗi N frame hiển thị (0 = tắt).")
    parser.add_argument("--save_dir", type=str, default="demo_debug")
    parser.add_argument("--max_seconds", type=float, default=None, help="Tự dừng sau N giây (debug/test).")
    parser.add_argument("--no_log", action="store_true", help="Không ghi fallen_events_log.csv.")
    parser.add_argument("--save_video", type=str, default=None, help="Xuất video đã overlay ra file mp4.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: fps={TARGET_FPS}, frames/window={NUM_FRAMES_PER_WINDOW}, "
          f"stride={STRIDE_SECONDS:.4f}s, num_features(classifier input)={NUM_FEATURES}")

    print("Loading VideoMAE-K400 extractor...")
    extractor = OnlineVideoMAEExtractor(device)
    print("Loading baseline FeatureTransformer (checkpoint-7000)...")
    baseline_model = load_baseline_model(device)

    need_hybrid = args.mode in ("hybrid", "both")
    hybrid_model = None
    if need_hybrid:
        print("Loading Hybrid v8 (LearnableConditioner, seed42)...")
        hybrid_model = load_hybrid_model(device)

    rule_conditioner = RuleBasedConditioner(boost_window_seconds=10.0, decay_halflife_seconds=4.0)

    is_video_file = args.video is not None
    cap = cv2.VideoCapture(args.video if is_video_file else args.camera_id)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được nguồn video: {args.video or args.camera_id}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = 30.0
    frame_step = max(int(round(native_fps / TARGET_FPS)), 1)
    print(f"Nguồn: {'video file ' + args.video if is_video_file else f'webcam {args.camera_id}'}, "
          f"native_fps={native_fps:.2f}, frame_step={frame_step} (lấy 1/{frame_step} frame -> ~{TARGET_FPS}fps)")

    gt_segments = load_ground_truth(args.video) if is_video_file else None
    if gt_segments:
        print("Có nhãn thật (ground truth) cho video này — sẽ hiển thị dòng GT.")
    writer = None

    save_dir = Path(args.save_dir)
    if args.save_every > 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    csv_writer = None
    if not args.no_log:
        is_new = not LOG_PATH.exists()
        log_file = open(LOG_PATH, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(log_file)
        if is_new:
            csv_writer.writerow(["timestamp", "elapsed_seconds", "mode", "label"])

    win16 = deque(maxlen=NUM_FRAMES_PER_WINDOW)   # resampled RGB frame cho 1 cửa sổ VideoMAE
    raw_features = deque(maxlen=NUM_FEATURES)     # 18 raw feature liên tiếp cho classifier
    history: list[dict] = []                      # {start_time, baseline_probs} cho conditioner

    prev_label = {"rule_based": None, "hybrid": None}

    frame_idx = 0
    raw_frame_counter = 0
    t0 = time.time()
    segment_count = 0
    segment_times: deque = deque(maxlen=30)  # để đo tốc độ xử lý segment/giây (cửa sổ trượt)
    last_lines: list[tuple[str, tuple[int, int, int]]] = [("dang khoi dong...", COLOR_GREEN)]
    any_fallen_current = False

    print("Bắt đầu xử lý. Nhấn 'q' để thoát (nếu có cửa sổ hiển thị)." if not args.no_display else
          "Chạy chế độ headless (--no_display).")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1

            if is_video_file:
                # Xử lý nhanh hết mức có thể khi test bằng file — dùng thời gian MÔ PHỎNG
                # theo frame_idx/native_fps, không chờ theo tốc độ thật của video.
                elapsed = frame_idx / native_fps
            else:
                elapsed = time.time() - t0

            if frame_idx % frame_step == 0:
                raw_frame_counter += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                win16.append(frame_rgb)

                if len(win16) == NUM_FRAMES_PER_WINDOW:
                    t_infer_start = time.time()
                    feat = extractor.extract(list(win16))
                    raw_features.append(feat)

                    if len(raw_features) == NUM_FEATURES:
                        feats_tensor = torch.tensor(
                            np.stack(list(raw_features)), dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        with torch.no_grad():
                            logits = baseline_model(feats_tensor)["logits"]
                            baseline_probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                        history.append({"start_time": elapsed, "baseline_probs": baseline_probs})
                        history = [h for h in history if elapsed - h["start_time"] <= HISTORY_WINDOW_SECONDS]

                        lines = []
                        any_fallen_current = False

                        if args.mode in ("rule_based", "both"):
                            rule_seq = rule_based_adjust(rule_conditioner, history, device)
                            rule_pred = int(rule_seq[-1].argmax())
                            color = status_color(rule_pred)
                            lines.append((f"Rule-based: {CLASS_NAMES_10[rule_pred]}", color))
                            if rule_pred == FALLEN_ID:
                                any_fallen_current = True
                                if prev_label["rule_based"] != "fallen" and csv_writer is not None:
                                    log_fallen_event(csv_writer, "rule_based", elapsed)
                            prev_label["rule_based"] = CLASS_NAMES_10[rule_pred]
                        else:
                            rule_seq = None

                        if args.mode in ("hybrid", "both"):
                            if rule_seq is None:
                                rule_seq = rule_based_adjust(rule_conditioner, history, device)
                            hybrid_seq = hybrid_adjust(hybrid_model, rule_seq, history, device)
                            hybrid_pred = int(hybrid_seq[-1].argmax())
                            color = status_color(hybrid_pred)
                            lines.append((f"Hybrid v8: {CLASS_NAMES_10[hybrid_pred]}", color))
                            if hybrid_pred == FALLEN_ID:
                                any_fallen_current = True
                                if prev_label["hybrid"] != "fallen" and csv_writer is not None:
                                    log_fallen_event(csv_writer, "hybrid", elapsed)
                            prev_label["hybrid"] = CLASS_NAMES_10[hybrid_pred]

                        last_lines = lines
                        segment_count += 1
                        segment_times.append(time.time() - t_infer_start)

            measured_fps = 1.0 / (sum(segment_times) / len(segment_times)) if segment_times else 0.0
            blink_on = int(time.time() * 2) % 2 == 0
            display_frame = draw_overlay(frame_bgr, last_lines, measured_fps, blink_on, any_fallen_current,
                                         gt_text=gt_at(gt_segments, elapsed))
            if args.save_video:
                if writer is None:
                    hh, ww = display_frame.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), native_fps, (ww, hh))
                writer.write(display_frame)

            if not args.no_display:
                cv2.imshow("OmniFall Realtime Demo", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.save_every > 0 and frame_idx % args.save_every == 0:
                cv2.imwrite(str(save_dir / f"frame_{frame_idx:06d}.png"), display_frame)

            if args.max_seconds is not None and elapsed >= args.max_seconds:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()

    total_time = time.time() - t0
    print(f"\n=== Kết thúc ===")
    print(f"Tổng số segment đã xử lý (classifier chạy): {segment_count}")
    if segment_times:
        avg_latency_ms = (sum(segment_times) / len(segment_times)) * 1000
        print(f"Độ trễ trung bình 1 lần cập nhật (VideoMAE + classifier + conditioner): {avg_latency_ms:.1f} ms")
        print(f"Tốc độ xử lý trung bình: {1000/avg_latency_ms:.2f} segment/giây")
    if not is_video_file:
        print(f"Thời gian chạy thực tế: {total_time:.1f}s")


if __name__ == "__main__":
    main()
