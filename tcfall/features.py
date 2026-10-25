"""Trích feature VideoMAE-K400 (mean-pool token cuối) cho cửa sổ trượt 16 frame, bước 1 frame.

Tiền xử lý ảnh dùng đúng VideoMAEImageProcessor (resize cạnh ngắn 224, center-crop 224,
chuẩn hoá ImageNet) như lúc tạo feature train baseline, để không lệch phân phối feature.
Mỗi frame chỉ được resize/normalize 1 lần rồi ghép cửa sổ bằng lập chỉ mục tensor.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

from tcfall.config import VIDEOMAE_NAME, WINDOW_FRAMES


class VideoMAEExtractor:
    def __init__(self, device: str | None = None, fp16: bool = True, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_NAME)
        self.model = VideoMAEModel.from_pretrained(VIDEOMAE_NAME).to(self.device).eval()
        self.fp16 = fp16 and self.device == "cuda"
        if self.fp16:
            self.model = self.model.half()
        self.batch_size = batch_size

    def preprocess(self, frames: list[np.ndarray], crop: str = "center") -> torch.Tensor:
        """list RGB uint8 -> (N, 3, 224, 224), xử lý theo khối 128 frame để tiết kiệm RAM.

        crop: "center" (mặc định của processor) | "start" (trái / trên) | "end" (phải / dưới)
        — dùng cho TTA không gian: resize cạnh ngắn 224 như processor, rồi cắt 224 ở đầu/cuối
        cạnh dài thay vì ở giữa.
        """
        out = []
        for i in range(0, len(frames), 128):
            chunk = frames[i:i + 128]
            if crop == "center":
                px = self.processor([chunk], return_tensors="pt")["pixel_values"][0]
            else:
                px = self.processor([chunk], return_tensors="pt", do_center_crop=False)["pixel_values"][0]
                h, w = px.shape[-2:]
                if w >= h:
                    px = px[..., :, :224] if crop == "start" else px[..., :, w - 224:]
                else:
                    px = px[..., :224, :] if crop == "start" else px[..., h - 224:, :]
            out.append(px.half() if self.fp16 else px)
        return torch.cat(out, 0)

    @torch.no_grad()
    def embed_windows(self, pixels: torch.Tensor) -> np.ndarray:
        """pixels (N, 3, H, W) -> (max(N-15, 1), 768): 1 feature cho mỗi cửa sổ 16 frame."""
        n = pixels.shape[0]
        if n < WINDOW_FRAMES:  # video quá ngắn: lặp frame cuối
            pixels = torch.cat([pixels, pixels[-1:].expand(WINDOW_FRAMES - n, -1, -1, -1)], 0)
            n = WINDOW_FRAMES
        starts = list(range(0, n - WINDOW_FRAMES + 1))
        feats = []
        for i in range(0, len(starts), self.batch_size):
            win = torch.stack([pixels[s:s + WINDOW_FRAMES] for s in starts[i:i + self.batch_size]])
            out = self.model(pixel_values=win.to(self.device))
            feats.append(out.last_hidden_state.mean(dim=1).float().cpu().numpy())
        return np.concatenate(feats, 0).astype(np.float32)

    def extract_variants(self, frames_by_variant: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        """Tiền xử lý mỗi frame duy nhất 1 lần (dùng chung giữa các biến thể)."""
        uniq: dict[int, int] = {}
        uniq_frames: list[np.ndarray] = []
        for frames in frames_by_variant.values():
            for f in frames:
                if id(f) not in uniq:
                    uniq[id(f)] = len(uniq_frames)
                    uniq_frames.append(f)
        px = self.preprocess(uniq_frames)
        result = {}
        for v, frames in frames_by_variant.items():
            idx = torch.tensor([uniq[id(f)] for f in frames], dtype=torch.long)
            result[v] = self.embed_windows(px[idx])
        return result
