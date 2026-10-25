"""Baseline classifier — tái lập cách paper OmniFall fine-tune VideoMAE-K400 cho 16-class."""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineClassifier(nn.Module):
    """
    Classifier head đơn giản trên đặc trưng đã trích xuất (VideoMAE/SigLIP2).
    Đây là baseline "mù thời gian" — mỗi segment được phân loại độc lập, giống
    cách paper OmniFall đánh giá trong Table 2-4 (đây chính là điểm mà module
    conditioning ở conditioning.py sẽ cải thiện).
    """

    def __init__(self, feature_dim: int, num_classes: int = 16,
                 hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, feature_dim) -> logits: (batch, num_classes)"""
        return self.net(features)


def compute_class_weights(labels: torch.Tensor, num_classes: int = 16) -> torch.Tensor:
    """
    Trọng số lớp nghịch đảo tần suất, để giảm ảnh hưởng của class imbalance
    (fallen/fall là lớp hiếm so với walk/standing) — paper cũng lưu ý vấn đề này
    khi giải thích macro-F1 thấp hơn balanced accuracy.
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights
