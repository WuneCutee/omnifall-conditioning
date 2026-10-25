"""
Bước 3v2 (hybrid) — Bước 1: cache logits của baseline đã train (checkpoint-3000) cho từng
segment, theo đúng thứ tự thời gian trong mỗi video, cho cả 3 split (of-syn train/val, OOPS
test). Dùng làm input cho LearnableConditioner thay vì feature VideoMAE thô — tận dụng baseline
mạnh (18 feature/segment, đã train kỹ) thay vì mean-pool 1 vector nghèo thông tin hơn.

Output: data/omnifall/baseline_logits/{split_name}.pkl
  {video_path: {"logits": (n_seg,10) float32, "labels": (n_seg,) int, "starts": (n_seg,) float32}}

Usage:
    python -m scripts.cache_baseline_logits
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / "omnifall-experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from safetensors.torch import load_file  # noqa: E402

from data.feature_dataset import FeatureVideoDataset  # noqa: E402
from models.feature_transformer import FeatureTransformer  # noqa: E402

OMNIFALL_ROOT = PROJECT_ROOT / "data" / "omnifall"
CHECKPOINT = EXPERIMENTS_ROOT / "outputs" / "2026-09-08" / "16-19-21" / \
    "feature_transformer_768_2l" / "checkpoint-3000" / "model.safetensors"
CACHE_DIR = OMNIFALL_ROOT / "baseline_logits"

NUM_FEATURES = 18
FEATURE_DIM = 768
HIDDEN_DIM = 768
NUM_CLASSES = 10
NUM_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 128


def load_model(device: str) -> FeatureTransformer:
    model = FeatureTransformer(
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_features=NUM_FEATURES,
    )
    state_dict = load_file(str(CHECKPOINT))
    state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def cache_split(model, device, dataset_name: str, mode: str, split: str, out_name: str):
    dataset = FeatureVideoDataset(
        feature_root=str(OMNIFALL_ROOT / dataset_name / "features" / "videomae-kinetics-7d5fps-1f"),
        annotations_file=str(OMNIFALL_ROOT / "labels" / "dataset" / f"{dataset_name}.csv"),
        feature_fps=7.5, feature_frames=16, feature_stride=0.133333333, feature_centered=False,
        feature_type="videomae-kinetics-7d5fps-1f", dataset_name=dataset_name, mode=mode,
        split=split, split_root=str(OMNIFALL_ROOT / "splits"), num_features=NUM_FEATURES,
        get_features=True,
    )
    print(f"{out_name}: {len(dataset)} segments, {len(dataset.video_to_indices)} videos")

    result = {}
    with torch.no_grad():
        for video_path, indices in tqdm(dataset.video_to_indices.items(), desc=out_name):
            segs = [dataset[i] for i in indices]  # đã sort theo start
            feats = torch.stack([s["features"] for s in segs]).to(device)  # (n_seg, 18, 768)
            labels = [int(s["label"]) for s in segs]
            starts = [float(s["start"]) for s in segs]

            logits_chunks = []
            for i in range(0, len(feats), BATCH_SIZE):
                chunk = feats[i:i + BATCH_SIZE]
                out = model(chunk)["logits"]
                logits_chunks.append(out.cpu())
            logits = torch.cat(logits_chunks, dim=0).numpy()

            result[video_path] = {"logits": logits, "labels": labels, "starts": starts}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{out_name}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"  -> saved {CACHE_DIR / f'{out_name}.pkl'}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(device)

    cache_split(model, device, "of-syn", "train", "cs", "of-syn_train")
    cache_split(model, device, "of-syn", "val", "cs", "of-syn_val")
    cache_split(model, device, "OOPS", "val", "cs", "OOPS_val")
    cache_split(model, device, "OOPS", "test", "cs", "OOPS_test")


if __name__ == "__main__":
    main()
