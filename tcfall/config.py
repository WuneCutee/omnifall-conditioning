"""Đường dẫn và hằng số dùng chung cho toàn bộ pipeline tcfall."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "omnifall"
EXPERIMENTS_ROOT = ROOT / "omnifall-experiments"
OUTPUTS = ROOT / "outputs_v2"          # mọi kết quả mới của pipeline v2 nằm ở đây
CKPT_DIR = ROOT / "checkpoints"

LABELS_RAW = DATA / "labels" / "dataset"
SPLITS = DATA / "splits"
OOPS_VIDEO = DATA / "OOPS" / "video"

# Backbone + baseline (giữ nguyên cấu hình đã train, Table 12 của OmniFall)
VIDEOMAE_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
TARGET_FPS = 7.5
WINDOW_FRAMES = 16                     # 1 cửa sổ VideoMAE = 16 frame = 2.13 s
FEATURE_STRIDE = 2 / 15                # 1 frame ở 7.5 fps
NUM_FEATURES = 18                      # classifier nhận 18 feature liên tiếp
FEATURE_DIM = 768
BASELINE_CKPT = (EXPERIMENTS_ROOT / "outputs" / "2026-09-08" / "21-55-48" /
                 "feature_transformer_768_2l" / "checkpoint-7000" / "model.safetensors")

CLASS_NAMES = ["walk", "fall", "fallen", "sit_down", "sitting",
               "lie_down", "lying", "stand_up", "standing", "other"]
NUM_CLASSES = len(CLASS_NAMES)
FALL, FALLEN, LIE_DOWN, LYING = 1, 2, 5, 6
TRANSIENT_TO_STATIC = {1: 2, 3: 4, 5: 6, 7: 8}

LABEL_FILES = {  # tên dataset -> file nhãn gốc
    "of-syn": "of-syn.csv", "gmdcsa24": "GMDCSA24.csv", "le2i": "le2i.csv",
    "caucafall": "caucafall.csv", "OOPS": "OOPS.csv",
}

# Hai biến thể feature OOPS: "legacy" tái tạo đúng cách lấy mẫu cũ (bước nguyên round(fps/7.5)),
# "timefix" lấy mẫu theo đồng hồ thời gian thật (đúng 7.5 fps với mọi fps gốc).
FEATURE_VARIANTS = ("legacy", "timefix")


def feature_dir(dataset: str, variant: str) -> Path:
    return DATA / dataset / "features_v2" / variant
