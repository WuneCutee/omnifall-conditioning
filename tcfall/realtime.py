"""Engine suy luận realtime (không phụ thuộc giao diện).

Luồng dữ liệu:
  frame (bất kỳ fps)  --lấy mẫu theo đồng hồ 7.5 fps-->  tiền xử lý GPU (224x224)
  -> bộ đệm 16 frame -> VideoMAE (fp16) -> 1 feature / frame mới
  -> bộ đệm 18 feature -> baseline -> p_t (10 lớp)
  -> [StreamConditioner] gom p_t thành "pseudo-segment" mỗi `segment_hop` giây,
     conditioner nhân quả (rule / hybrid / hmm), bộ báo động có trễ (hysteresis).

Lý do gom pseudo-segment: conditioner được huấn luyện trên chuỗi segment cách nhau ~1.5 s;
đưa vào từng bước 0.13 s sẽ lệch phân phối (đã ghi nhận ở bản demo cũ).
StreamConditioner không cần GPU/VideoMAE nên dùng được để mô phỏng streaming offline
trên feature đã trích (pipeline/s07_stream_eval.py).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from tcfall.config import CLASS_NAMES, FALL, FALLEN, NUM_FEATURES, TARGET_FPS, WINDOW_FRAMES
from tcfall.conditioners import (DEFAULT_HYBRID, hmm_filter, hybrid_apply, load_hybrid, prior_shift,
                                 rule_condition)

MODES = ("baseline", "rule", "hybrid", "hmm")


@dataclass
class EngineConfig:
    mode: str = "rule"
    segment_hop: float = 1.0          # s, chu kỳ chốt pseudo-segment cho conditioner
    history_seconds: float = 20.0
    tau: float = 4.0
    window: float = 10.0
    hybrid_ckpt: str = str(DEFAULT_HYBRID)
    hmm_A: np.ndarray | None = None   # ma trận chuyển tiếp (từ outputs_v2/best_config.json)
    hmm_prior: np.ndarray | None = None
    hmm_strength: float = 1.0
    src_prior: np.ndarray | None = None   # hiệu chỉnh label shift (nếu best_config chọn)
    tgt_prior: np.ndarray | None = None
    prior_alpha: float = 0.0
    alarm_on_seconds: float = 1.5     # fallen liên tục >= x giây mới báo động
    alarm_off_seconds: float = 3.0    # hết fallen >= x giây mới tắt báo động
    max_batch: int = 8                # số cửa sổ VideoMAE tối đa xử lý 1 lượt khi bị trễ
    spatial_tta: bool = False         # trung bình 3 crop không gian (trái/giữa/phải) — tốn x3 GPU


@dataclass
class EngineState:
    t: float = 0.0
    baseline_probs: np.ndarray = field(default_factory=lambda: np.full(10, 0.1))
    probs: np.ndarray = field(default_factory=lambda: np.full(10, 0.1))
    label: str = "warmup"
    ready: bool = False
    alarm: bool = False
    alarm_events: list = field(default_factory=list)
    timeline: deque = field(default_factory=lambda: deque(maxlen=400))   # (t, class_id)
    latency_ms: float = 0.0


class StreamConditioner:
    """Nhận p_t (xác suất baseline tại thời điểm t) theo dòng, trả xác suất đã điều kiện hoá."""

    def __init__(self, cfg: EngineConfig, hybrid_model=None):
        self.cfg = cfg
        self.hybrid = hybrid_model
        self.reset()

    def reset(self):
        self.step_probs: list[np.ndarray] = []
        self.segments: list[tuple[float, np.ndarray]] = []
        self.last_seg_t = None
        self.fallen_since = None
        self.clear_since = None
        self.state = EngineState()

    def update(self, t: float, p: np.ndarray) -> EngineState:
        c, st = self.cfg, self.state
        if c.prior_alpha > 0 and c.src_prior is not None:
            p = prior_shift(p[None], c.src_prior, c.tgt_prior, c.prior_alpha)[0]
        st.t, st.baseline_probs, st.ready = t, p, True
        self.step_probs.append(p)
        if self.last_seg_t is None:
            self.last_seg_t = t
        seg_p = np.mean(self.step_probs, 0)
        if t - self.last_seg_t >= c.segment_hop:
            # chốt pseudo-segment hiện tại vào lịch sử
            self.segments.append((self.last_seg_t, seg_p))
            self.segments = [s for s in self.segments if t - s[0] <= c.history_seconds]
            self.step_probs, self.last_seg_t = [], t
            st.probs = self._condition(self.segments)
        elif c.mode == "baseline":
            st.probs = p
        else:
            # cập nhật tức thời: lịch sử đã chốt + segment tạm (chưa chốt) đang hình thành
            st.probs = self._condition(self.segments + [(self.last_seg_t, seg_p)])
        k = int(st.probs.argmax())
        st.label = CLASS_NAMES[k]
        st.timeline.append((t, k))
        self._update_alarm(t, k)
        return st

    def _condition(self, segments) -> np.ndarray:
        starts = np.array([s for s, _ in segments])
        P = np.stack([p for _, p in segments])
        mode = self.cfg.mode
        if mode == "baseline":
            return P[-1]
        if mode == "hmm" and self.cfg.hmm_A is not None:
            return hmm_filter(P, self.cfg.hmm_A, self.cfg.hmm_prior, self.cfg.hmm_strength)[-1]
        R = rule_condition(P, starts, self.cfg.tau, self.cfg.window)
        if mode == "hybrid" and self.hybrid is not None:
            return hybrid_apply(self.hybrid, R, starts)[-1]
        return R[-1]

    def _update_alarm(self, t: float, k: int):
        st, c = self.state, self.cfg
        if k == FALLEN:
            self.clear_since = None
            if self.fallen_since is None:
                self.fallen_since = t
            if not st.alarm and t - self.fallen_since >= c.alarm_on_seconds:
                st.alarm = True
                st.alarm_events.append(t)
        else:
            self.fallen_since = None
            if st.alarm:
                if self.clear_since is None:
                    self.clear_since = t
                if t - self.clear_since >= c.alarm_off_seconds:
                    st.alarm = False


class FallEngine:
    """Ghép VideoMAE + baseline + StreamConditioner cho đầu vào là frame thô."""

    def __init__(self, cfg: EngineConfig, device: str | None = None):
        from transformers import VideoMAEModel

        from tcfall.baseline import load_baseline
        from tcfall.config import VIDEOMAE_NAME
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = self.device == "cuda"
        vm = VideoMAEModel.from_pretrained(VIDEOMAE_NAME).to(self.device).eval()
        self.videomae = vm.half() if self.fp16 else vm
        self.baseline = load_baseline(self.device)
        self.cond = StreamConditioner(cfg, load_hybrid(cfg.hybrid_ckpt, "cpu"))
        self.reset()

    @property
    def crops(self) -> tuple[str, ...]:
        return ("center", "start", "end") if self.cfg.spatial_tta else ("center",)

    def reset(self):
        self.next_sample_t = 0.0
        self.frames = deque(maxlen=WINDOW_FRAMES)            # tensor (K,3,224,224) trên GPU, K = số crop
        self.pending: list[tuple[float, torch.Tensor]] = []  # (t, cửa sổ (K,16,3,224,224)) chờ embed
        self.feats = deque(maxlen=NUM_FEATURES)              # tensor (K, 768)
        self.cond.reset()

    @property
    def state(self) -> EngineState:
        return self.cond.state

    def push_frame(self, frame_rgb: np.ndarray, t: float) -> bool:
        """Đưa 1 frame với mốc thời gian t (giây). Trả về True nếu frame được lấy mẫu."""
        from tcfall.preprocess_gpu import preprocess_frames
        if t + 1e-6 < self.next_sample_t:
            return False
        while self.next_sample_t <= t:   # nguồn bị giật: bỏ các mốc đã lỡ thay vì dồn frame
            self.next_sample_t += 1.0 / TARGET_FPS
        px = preprocess_frames([frame_rgb], self.device, self.fp16, crops=self.crops)
        self.frames.append(px[:, 0] if px.dim() == 5 else px)      # (K,3,224,224)
        if len(self.frames) == WINDOW_FRAMES:
            self.pending.append((t, torch.stack(list(self.frames), 1)))  # (K,16,3,224,224)
            if len(self.pending) > 4 * self.cfg.max_batch:   # GPU không kịp: giữ cửa sổ mới nhất
                self.pending = self.pending[-self.cfg.max_batch:]
        return True

    @torch.no_grad()
    def step(self) -> EngineState:
        """Xử lý các cửa sổ đang chờ (gom batch). Gọi liên tục từ luồng suy luận."""
        if not self.pending:
            return self.state
        t0 = time.perf_counter()
        batch, self.pending = self.pending[: self.cfg.max_batch], self.pending[self.cfg.max_batch:]
        x = torch.cat([w for _, w in batch])                    # (B*K,16,3,224,224)
        k = len(self.crops)
        f = self.videomae(pixel_values=x).last_hidden_state.mean(1).float().view(len(batch), k, -1)
        for (t, _), feat in zip(batch, f):
            self.feats.append(feat)                             # (K,768)
            if len(self.feats) == NUM_FEATURES:
                seq = torch.stack(list(self.feats), 1)          # (K,18,768)
                probs = torch.softmax(self.baseline(seq)["logits"].float(), -1).mean(0)
                self.cond.update(t, probs.cpu().numpy())
        self.state.latency_ms = (time.perf_counter() - t0) * 1000 / len(batch)
        return self.state

    @property
    def status(self) -> str:
        st = self.state
        if not st.ready:
            return "WARMUP"
        if st.alarm:
            return "ALARM"
        k = int(st.probs.argmax())
        return "FALLEN" if k == FALLEN else "FALL" if k == FALL else "NORMAL"
