"""
Bước 2 — Pilot: áp RuleBasedConditioner (src/conditioning.py) POST-HOC lên output xác suất
của baseline đã train ở Bước 1 (checkpoint chính thức từ omnifall-experiments), KHÔNG train
lại gì. So sánh Fallen-F1 trước/sau conditioning trên cùng 1 tập test OF-ItW (OOPS).

Cách làm:
  1. Load model FeatureTransformer + checkpoint tốt nhất (theo eval_OOPS_fall_f1) từ Bước 1.
  2. Load FeatureVideoDataset (code chính thức) cho OOPS test split — segments mỗi video đã
     được sort theo thời gian sẵn (xem data/feature_dataset.py).
  3. Với mỗi video: chạy model lấy softmax probs cho từng segment (theo đúng thứ tự thời gian),
     ghép thành (1, seq_len, 10), rồi áp RuleBasedConditioner.apply().
  4. Tính lại Fallen-F1 (và Fall-F1, Balanced Acc) cho cả 2 trường hợp, in bảng so sánh.

Usage:
    python -m scripts.pilot_rule_conditioning
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / "omnifall-experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from safetensors.torch import load_file  # noqa: E402

from data.feature_dataset import FeatureVideoDataset  # noqa: E402
from models.feature_transformer import FeatureTransformer  # noqa: E402

from src.conditioning import RuleBasedConditioner  # noqa: E402
from src.utils import compute_all_metrics, print_metrics_table  # noqa: E402

OMNIFALL_ROOT = PROJECT_ROOT / "data" / "omnifall"
CHECKPOINT = EXPERIMENTS_ROOT / "outputs" / "2026-09-08" / "16-19-21" / \
    "feature_transformer_768_2l" / "checkpoint-3000" / "model.safetensors"

NUM_FEATURES = 18
FEATURE_DIM = 768
HIDDEN_DIM = 768
NUM_CLASSES = 10
NUM_LAYERS = 2
NUM_HEADS = 8
FALL_CLASS_ID = 1
FALLEN_CLASS_ID = 2


def load_model(device: str) -> FeatureTransformer:
    model = FeatureTransformer(
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_features=NUM_FEATURES,
    )
    state_dict = load_file(str(CHECKPOINT))
    # Checkpoint được lưu qua FeatureTransformerWrapper (self.model = FeatureTransformer(...)),
    # nên mọi key có tiền tố "model." cần bỏ đi trước khi load vào FeatureTransformer trần.
    state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"[WARN] missing keys: {missing}, unexpected keys: {unexpected}")
    return model.to(device).eval()


def load_oops_test_dataset() -> FeatureVideoDataset:
    return FeatureVideoDataset(
        feature_root=str(OMNIFALL_ROOT / "OOPS" / "features" / "videomae-kinetics-7d5fps-1f"),
        annotations_file=str(OMNIFALL_ROOT / "labels" / "dataset" / "OOPS.csv"),
        feature_fps=7.5,
        feature_frames=16,
        feature_stride=0.133333333,
        feature_centered=False,
        feature_type="videomae-kinetics-7d5fps-1f",
        dataset_name="OOPS",
        mode="test",
        split="cs",
        split_root=str(OMNIFALL_ROOT / "splits"),
        num_features=NUM_FEATURES,
        get_features=True,
    )


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(device)
    dataset = load_oops_test_dataset()
    print(f"Loaded OOPS test dataset: {len(dataset)} segments, {len(dataset.video_to_indices)} videos")

    conditioner = RuleBasedConditioner(boost_window_seconds=10.0, decay_halflife_seconds=4.0)

    all_true, all_pred_baseline, all_pred_conditioned = [], [], []

    for video_path, indices in dataset.video_to_indices.items():
        segs = [dataset[i] for i in indices]  # đã sort theo thời gian (xem feature_dataset.py)

        feats = torch.stack([s["features"] for s in segs]).unsqueeze(0).to(device)  # (1, seq, 18, 768)
        labels = torch.tensor([s["label"] for s in segs])
        starts = torch.tensor([[s["start"] for s in segs]], dtype=torch.float)  # (1, seq)
        mask = torch.ones(1, len(segs), dtype=torch.bool)

        batch, seq_len, nf, fd = feats.shape
        logits = model(feats.reshape(batch * seq_len, nf, fd))["logits"].reshape(batch, seq_len, NUM_CLASSES)
        probs = F.softmax(logits, dim=-1).cpu()

        adjusted = conditioner.apply(probs.clone(), starts, mask)

        all_true.extend(labels.numpy())
        all_pred_baseline.extend(probs[0].argmax(dim=-1).numpy())
        all_pred_conditioned.extend(adjusted[0].argmax(dim=-1).numpy())

    y_true = np.array(all_true)
    y_pred_base = np.array(all_pred_baseline)
    y_pred_cond = np.array(all_pred_conditioned)

    print(f"\nTổng số segment đánh giá: {len(y_true)}")

    print("\n=== BASELINE (chưa conditioning) ===")
    m_base = compute_all_metrics(y_true, y_pred_base, FALL_CLASS_ID, FALLEN_CLASS_ID, num_classes=NUM_CLASSES)
    print_metrics_table(m_base)

    print("\n=== RULE-BASED CONDITIONING (post-hoc) ===")
    m_cond = compute_all_metrics(y_true, y_pred_cond, FALL_CLASS_ID, FALLEN_CLASS_ID, num_classes=NUM_CLASSES)
    print_metrics_table(m_cond)

    delta = (m_cond["fallen"]["f1"] - m_base["fallen"]["f1"]) * 100
    print(f"\n>>> Thay đổi Fallen-F1: {delta:+.2f} điểm phần trăm "
          f"({m_base['fallen']['f1']*100:.2f}% -> {m_cond['fallen']['f1']*100:.2f}%)")


if __name__ == "__main__":
    main()
