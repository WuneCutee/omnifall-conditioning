"""Nạp classifier baseline (FeatureTransformer, checkpoint-7000 của repo OmniFall chính thức)."""
from __future__ import annotations

import sys

import torch
from safetensors.torch import load_file

from tcfall.config import BASELINE_CKPT, EXPERIMENTS_ROOT, FEATURE_DIM, NUM_CLASSES, NUM_FEATURES

if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
from models.feature_transformer import FeatureTransformer  # noqa: E402


def load_baseline(device: str = "cuda", ckpt: str | None = None) -> FeatureTransformer:
    """Checkpoint gốc (St+Sy). `ckpt` (hoặc biến môi trường TCFALL_BASELINE_CKPT) = file .pt
    state_dict của bản đã fine-tune trên ItW-train (pipeline/s05b_finetune_itw.py)."""
    import os
    model = FeatureTransformer(feature_dim=FEATURE_DIM, hidden_dim=768, num_classes=NUM_CLASSES,
                               num_layers=2, num_heads=8, num_features=NUM_FEATURES)
    ckpt = ckpt or os.environ.get("TCFALL_BASELINE_CKPT")
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    else:
        sd = load_file(str(BASELINE_CKPT))
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_logits(model, seqs: torch.Tensor, batch: int = 256) -> torch.Tensor:
    """seqs (N, 18, 768) -> logits (N, 10)."""
    dev = next(model.parameters()).device
    out = [model(seqs[i:i + batch].to(dev).float())["logits"].float().cpu() for i in range(0, len(seqs), batch)]
    return torch.cat(out, 0)
