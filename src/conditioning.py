"""
Module cốt lõi của đóng góp: Transition-Conditioned State Classification.

Ý tưởng: OmniFall taxonomy định nghĩa fallen/lying là trạng thái TĨNH theo sau
hành động CHUYỂN TIẾP (fall/lie_down), nhưng baseline phân loại từng segment độc
lập, không tận dụng thông tin "hành động gì vừa xảy ra trước đó". Module này
lan truyền tín hiệu từ hành động chuyển tiếp sang các segment tĩnh theo sau.

Có 2 chế độ:
  1. RuleBasedConditioner — bản pilot đơn giản (Bước 2 trong kế hoạch), giống logic
     ElderFallGuard: khi phát hiện fall/lie_down, boost xác suất fallen/lying tương ứng
     trong N giây tiếp theo với suy giảm theo thời gian. Dùng để kiểm tra nhanh hypothesis
     trước khi đầu tư bản learnable.
  2. LearnableConditioner — bản đầy đủ (Bước 3), dùng cross-attention để model tự học
     cách trọng số hoá thông tin từ các segment trước đó, thay vì rule cứng.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import TRANSIENT_TO_STATIC, CLASS_TO_ID


class RuleBasedConditioner:
    """
    Bản pilot rule-based — KHÔNG có tham số học được, chỉ áp dụng post-hoc lên
    xác suất dự đoán của baseline model. Dùng để kiểm tra nhanh hypothesis
    (Bước 2 trong kế hoạch) trước khi đầu tư vào bản learnable.
    """

    def __init__(self, boost_window_seconds: float = 10.0,
                 decay_halflife_seconds: float = 4.0):
        self.boost_window_seconds = boost_window_seconds
        self.decay_halflife_seconds = decay_halflife_seconds

    def _decay_weight(self, elapsed_seconds: float) -> float:
        if elapsed_seconds > self.boost_window_seconds:
            return 0.0
        return 0.5 ** (elapsed_seconds / self.decay_halflife_seconds)

    def apply(self, probs: torch.Tensor, start_times: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        """
        probs: (batch, seq_len, num_classes) — xác suất softmax từ baseline model
        start_times: (batch, seq_len) — thời điểm bắt đầu mỗi segment, ĐƠN VỊ GIÂY
            (annotation OmniFall thật cho start/end bằng giây, không phải frame index —
            xem data/omnifall/STRUCTURE.md — nên không cần quy đổi qua fps nữa)
        mask: (batch, seq_len) — True nếu segment hợp lệ (không phải padding)

        Trả về: probs đã được điều chỉnh (cùng shape).
        """
        batch_size, seq_len, num_classes = probs.shape
        adjusted = probs.clone()

        for b in range(batch_size):
            valid_len = int(mask[b].sum().item())
            last_transient: dict[int, float] = {}  # {static_class_id: elapsed_time_since_transient}

            for t in range(valid_len):
                current_time = start_times[b, t].item()
                pred_class = int(probs[b, t].argmax().item())

                # Áp dụng boost cho mọi static class đã ghi nhớ TỪ TRƯỚC — dùng trạng thái
                # last_transient TRƯỚC KHI xét segment hiện tại, để chính segment "fall"/
                # "lie_down" vừa được phát hiện không tự ghi đè lên dự đoán của chính nó
                # (nếu không, elapsed=0 -> weight=1.0 -> mọi dự đoán "fall" biến thành "fallen"
                # ngay lập tức, xoá sổ class "fall" khỏi kết quả).
                #
                # QUAN TRỌNG: chỉ boost nếu segment hiện tại KHÔNG tự tin là 1 hành động
                # chuyển tiếp khác (vd chính nó cũng là "fall"). Nếu không có điều kiện này,
                # 1 hành động fall kéo dài nhiều segment liên tiếp (rất phổ biến — median
                # cách nhau ~1.2s trong dữ liệu thật) sẽ bị segment fall ĐẦU TIÊN "ăn theo"
                # đè segment fall THỨ HAI thành "fallen", phá Fall-F1 (đã verify bằng số:
                # 85% số segment true=fall bị đổi sai là đổi thành "fallen", 98% trong số đó
                # có 1 fall-prediction trước đó trong vòng 10s).
                if pred_class not in TRANSIENT_TO_STATIC:
                    for static_class, transient_time in list(last_transient.items()):
                        elapsed = current_time - transient_time
                        weight = self._decay_weight(elapsed)
                        if weight <= 0:
                            del last_transient[static_class]
                            continue
                        # Boost có trọng số: pha trộn xác suất gốc với prior từ transition
                        prior = torch.zeros(num_classes, device=probs.device)
                        prior[static_class] = 1.0
                        adjusted[b, t] = (1 - weight) * adjusted[b, t] + weight * prior

                # Nếu segment hiện tại được dự đoán là 1 hành động chuyển tiếp,
                # "ghi nhớ" nó để boost trạng thái tĩnh tương ứng ở các bước SAU (không phải
                # chính nó — xem chú thích ở trên).
                if pred_class in TRANSIENT_TO_STATIC:
                    static_target = TRANSIENT_TO_STATIC[pred_class]
                    last_transient[static_target] = current_time

        # Re-normalize để vẫn là phân phối xác suất hợp lệ
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return adjusted


class LearnableConditioner(nn.Module):
    """
    Bản đầy đủ: dùng self-attention theo thời gian để model tự học cách kết hợp
    thông tin từ các segment trước đó (thay vì rule cứng). Đây là phần đóng góp
    kỹ thuật chính của dự án — thay thế RuleBasedConditioner sau khi pilot xác nhận
    hypothesis có tín hiệu tích cực.
    """

    def __init__(self, feature_dim: int, num_classes: int = 16,
                 hidden_dim: int = 128, num_heads: int = 4,
                 context_window_seconds: float = 15.0, residual: bool = False,
                 dropout: float = 0.0):
        """
        residual: khi True, input được coi là LOGITS của 1 classifier khác đã train sẵn
            (vd baseline per-segment, không có context chéo segment) — feature_dim PHẢI
            bằng num_classes. Module này khi đó chỉ học phần ĐIỀU CHỈNH (delta) cộng vào
            logits gốc, thay vì học lại phân loại từ đầu trên representation nghèo hơn.
            Init classifier gần 0 để lúc bắt đầu train, output ~ logits gốc (không tệ hơn
            baseline), rồi học dần cách dùng ngữ cảnh thời gian để chỉnh cho tốt hơn.
        """
        super().__init__()
        self.context_window_seconds = context_window_seconds
        self.residual = residual
        if residual and feature_dim != num_classes:
            raise ValueError("residual=True yêu cầu feature_dim == num_classes (input phải là logits).")

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos_encoding = SinusoidalTimeEncoding(hidden_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        if residual:
            # Khởi tạo gần 0 để delta ban đầu ~ 0 -> output ban đầu ~ logits gốc của baseline.
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def _build_time_mask(self, start_times: torch.Tensor,
                          key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Chỉ cho phép attention nhìn về QUÁ KHỨ trong phạm vi context_window_seconds
        (causal + windowed attention) — mô phỏng đúng ràng buộc thực tế: hệ thống
        chỉ biết được các sự kiện ĐÃ xảy ra, không biết tương lai.
        """
        batch_size, seq_len = start_times.shape
        times = start_times.float()  # (batch, seq_len), đơn vị giây

        time_diff = times.unsqueeze(2) - times.unsqueeze(1)  # (batch, seq_len, seq_len): query - key
        causal = time_diff >= 0  # chỉ nhìn về quá khứ hoặc hiện tại
        within_window = time_diff <= self.context_window_seconds
        allowed = causal & within_window

        # Kết hợp với padding mask (loại bỏ segment padding)
        pad_allowed = key_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
        allowed = allowed & pad_allowed

        # torch.nn.MultiheadAttention cần attn_mask dạng additive (True = bị chặn -> -inf)
        attn_mask = ~allowed
        # Mở rộng cho multi-head: (batch * num_heads, seq_len, seq_len) sẽ được xử lý ở forward
        return attn_mask

    def forward(self, features: torch.Tensor, start_times: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        features: (batch, seq_len, feature_dim)
        start_times: (batch, seq_len) — thời điểm bắt đầu mỗi segment, ĐƠN VỊ GIÂY
        mask: (batch, seq_len) — True = segment hợp lệ

        Trả về logits: (batch, seq_len, num_classes)
        """
        batch_size, seq_len, _ = features.shape

        x = self.input_proj(features)
        x = x + self.pos_encoding(start_times)

        attn_mask = self._build_time_mask(start_times, mask)  # (batch, seq_len, seq_len)
        # Lặp lại cho từng head vì nn.MultiheadAttention với batch_first cần (batch*heads, L, L)
        num_heads = self.self_attn.num_heads
        attn_mask_expanded = attn_mask.repeat_interleave(num_heads, dim=0)

        key_padding_mask = ~mask  # True = bị bỏ qua

        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask_expanded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.attn_dropout(attn_out))
        x = self.norm2(x + self.ffn(x))

        logits = self.classifier(x)
        if self.residual:
            logits = logits + features  # học phần điều chỉnh (delta) cộng lên logits gốc
        return logits


class SinusoidalTimeEncoding(nn.Module):
    """Positional encoding dựa trên thời gian thực (giây), thay vì chỉ số thứ tự segment,
    để model biết được khoảng cách thời gian thực giữa các segment (video có tốc độ khác nhau)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, start_times: torch.Tensor) -> torch.Tensor:
        times = start_times.float()  # (batch, seq_len), đơn vị giây
        device = start_times.device
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(*times.shape, self.dim, device=device)
        pe[..., 0::2] = torch.sin(times.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(times.unsqueeze(-1) * div_term)
        return pe
