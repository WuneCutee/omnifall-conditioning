"""
Giống cache_baseline_logits.py nhưng cho baseline St(GMDCSA24+Le2i+CAUCAFall)+Sy mới —
gộp logits từ cả 4 dataset train nguồn vào 1 cache "stsyn_train"/"stsyn_val", cùng checkpoint
mới (outputs/2026-09-08/21-55-48/.../checkpoint-7000), để feed vào rule-based/hybrid v4.

Usage:
    python -m scripts.cache_baseline_logits_stsyn
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / "omnifall-experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from safetensors.torch import load_file  # noqa: E402

from data.feature_dataset import FeatureVideoDataset  # noqa: E402
from models.feature_transformer import FeatureTransformer  # noqa: E402

OMNIFALL_ROOT = PROJECT_ROOT / "data" / "omnifall"
CHECKPOINT = EXPERIMENTS_ROOT / "outputs" / "2026-09-08" / "23-33-34" / \
    "feature_transformer_768_2l" / "checkpoint-5000" / "model.safetensors"
CACHE_DIR = OMNIFALL_ROOT / "baseline_logits"

NUM_FEATURES = 18
FEATURE_DIM = 768
HIDDEN_DIM = 768
NUM_CLASSES = 10
NUM_LAYERS = 2
NUM_HEADS = 8
BATCH_SIZE = 128

STAGED_DATASETS = [
    ("of-syn", "of-syn.csv"),
    ("gmdcsa24", "GMDCSA24.csv"),
    ("le2i", "le2i.csv"),
    ("caucafall", "caucafall.csv"),
    ("mcfd", "mcfd.csv"),
]


def load_model(device: str) -> FeatureTransformer:
    model = FeatureTransformer(
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_features=NUM_FEATURES,
    )
    state_dict = load_file(str(CHECKPOINT))
    state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def cache_one_dataset(model, device, dataset_name: str, labels_file: str, mode: str) -> dict:
    dataset = FeatureVideoDataset(
        feature_root=str(OMNIFALL_ROOT / dataset_name / "features" / "videomae-kinetics-7d5fps-1f"),
        annotations_file=str(OMNIFALL_ROOT / "labels" / "dataset" / labels_file),
        feature_fps=7.5, feature_frames=16, feature_stride=0.133333333, feature_centered=False,
        feature_type="videomae-kinetics-7d5fps-1f", dataset_name=dataset_name, mode=mode,
        split="cs", split_root=str(OMNIFALL_ROOT / "splits"), num_features=NUM_FEATURES,
        get_features=True,
    )
    result = {}
    for video_path, indices in tqdm(dataset.video_to_indices.items(), desc=f"{dataset_name}/{mode}"):
        segs = [dataset[i] for i in indices]
        feats = torch.stack([s["features"] for s in segs]).to(device)
        labels = [int(s["label"]) for s in segs]
        starts = [float(s["start"]) for s in segs]

        logits_chunks = []
        for i in range(0, len(feats), BATCH_SIZE):
            chunk = feats[i:i + BATCH_SIZE]
            out = model(chunk)["logits"]
            logits_chunks.append(out.cpu())
        logits = torch.cat(logits_chunks, dim=0).numpy()
        # Khoá theo (dataset, path) để tránh trùng path giữa các dataset khác nhau.
        result[f"{dataset_name}::{video_path}"] = {"logits": logits, "labels": labels, "starts": starts}
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(device)

    for mode, out_name in [("train", "stsyn2_train"), ("val", "stsyn2_val")]:
        combined = {}
        for dataset_name, labels_file in STAGED_DATASETS:
            combined.update(cache_one_dataset(model, device, dataset_name, labels_file, mode))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"{out_name}.pkl", "wb") as f:
            pickle.dump(combined, f)
        print(f"{out_name}: {len(combined)} videos -> {CACHE_DIR / f'{out_name}.pkl'}")

    # OOPS val/test dùng checkpoint MỚI (khác checkpoint cũ đã cache trước đó).
    for mode, out_name in [("val", "OOPS_val_stsyn2"), ("test", "OOPS_test_stsyn2")]:
        result = cache_one_dataset(model, device, "OOPS", "OOPS.csv", mode)
        with open(CACHE_DIR / f"{out_name}.pkl", "wb") as f:
            pickle.dump(result, f)
        print(f"{out_name}: {len(result)} videos -> {CACHE_DIR / f'{out_name}.pkl'}")


if __name__ == "__main__":
    main()
