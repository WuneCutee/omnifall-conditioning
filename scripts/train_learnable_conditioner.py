"""
Bước 3 — Train LearnableConditioner (src/conditioning.py): thay RuleBasedConditioner bằng
self-attention theo thời gian, học cách kết hợp thông tin giữa các segment trong cùng 1 video
thay vì rule cứng. Train trên OF-Syn (train split), eval trên OF-ItW (OOPS, test split) —
cùng protocol Sy->ItW như Bước 1/2, cùng nhãn 10-lớp đã remap (labels/dataset/*.csv).

Input mỗi segment: mean-pool feature VideoMAE trong khoảng [start,end) của segment đó (dùng
lại công thức time_to_feature_index trong src/data.py) — khác baseline chính thức (dùng 18
feature liên tiếp không mean-pool), vì LearnableConditioner cần 1 vector/segment để làm
self-attention CHÉO SEGMENT (baseline không có attention chéo segment).

Usage:
    python -m scripts.train_learnable_conditioner
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner  # noqa: E402
from src.data import stride_frames_from_seconds, time_to_feature_index  # noqa: E402
from src.utils import compute_all_metrics, print_metrics_table, set_seed  # noqa: E402

OMNIFALL_ROOT = PROJECT_ROOT / "data" / "omnifall"
FPS = 7.5
STRIDE_SECONDS = 2 / 15
NUM_CLASSES = 10
FEATURE_DIM = 768
HIDDEN_DIM = 128
NUM_HEADS = 4
CONTEXT_WINDOW_SECONDS = 15.0
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4


def build_video_index(labels_csv: Path, split_csv: Path) -> dict[str, list[dict]]:
    """Trả về {path: [segment_dict, ...]} đã sort theo start, chỉ giữ video có trong split."""
    labels = pd.read_csv(labels_csv)
    split_paths = set(pd.read_csv(split_csv)["path"].tolist())

    videos: dict[str, list[dict]] = {}
    for row in labels.itertuples(index=False):
        if row.path not in split_paths:
            continue
        videos.setdefault(row.path, []).append(
            {"start": float(row.start), "end": float(row.end), "label": int(row.label)}
        )
    for path in videos:
        videos[path].sort(key=lambda s: s["start"])
    return videos


class VideoSequenceDataset(Dataset):
    """1 item = 1 video = chuỗi segment (feature mean-pool, label, start_time)."""

    def __init__(self, feature_root: Path, videos: dict[str, list[dict]]):
        self.feature_root = feature_root
        self.paths = list(videos.keys())
        self.videos = videos
        self.stride_frames = stride_frames_from_seconds(FPS, STRIDE_SECONDS)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        segs = self.videos[path]
        h5_path = self.feature_root / f"{path}.h5"
        with h5py.File(h5_path, "r") as f:
            feats_full = f["features"][:]  # (T, 768)

        feats, labels, starts = [], [], []
        for seg in segs:
            i0 = time_to_feature_index(seg["start"], FPS, self.stride_frames)
            i1 = time_to_feature_index(seg["end"], FPS, self.stride_frames)
            i0 = max(0, min(i0, len(feats_full) - 1))
            i1 = max(i0 + 1, min(i1, len(feats_full)))
            feats.append(feats_full[i0:i1].mean(axis=0))
            labels.append(seg["label"])
            starts.append(seg["start"])

        return {
            "features": torch.from_numpy(np.stack(feats)).float(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "start_times": torch.tensor(starts, dtype=torch.float),
        }


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]

    padded_feats = torch.zeros(len(batch), max_len, feat_dim)
    padded_labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    padded_starts = torch.zeros(len(batch), max_len, dtype=torch.float)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        padded_feats[i, :n] = item["features"]
        padded_labels[i, :n] = item["labels"]
        padded_starts[i, :n] = item["start_times"]
        mask[i, :n] = True

    return {"features": padded_feats, "labels": padded_labels, "start_times": padded_starts, "mask": mask}


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        feats = batch["features"].to(device)
        starts = batch["start_times"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"]

        logits = model(feats, starts, mask)
        preds = logits.argmax(dim=-1).cpu()

        valid = mask.cpu()
        all_true.extend(labels[valid].numpy())
        all_pred.extend(preds[valid].numpy())
    return np.array(all_true), np.array(all_pred)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    of_syn_root = OMNIFALL_ROOT / "of-syn" / "features" / "videomae-kinetics-7d5fps-1f"
    oops_root = OMNIFALL_ROOT / "OOPS" / "features" / "videomae-kinetics-7d5fps-1f"
    labels_dir = OMNIFALL_ROOT / "labels" / "dataset"
    splits_dir = OMNIFALL_ROOT / "splits" / "cs"

    train_videos = build_video_index(labels_dir / "of-syn.csv", splits_dir / "of-syn" / "train.csv")
    val_videos = build_video_index(labels_dir / "of-syn.csv", splits_dir / "of-syn" / "val.csv")
    test_videos = build_video_index(labels_dir / "OOPS.csv", splits_dir / "OOPS" / "test.csv")

    train_ds = VideoSequenceDataset(of_syn_root, train_videos)
    val_ds = VideoSequenceDataset(of_syn_root, val_videos)
    test_ds = VideoSequenceDataset(oops_root, test_videos)
    print(f"train videos: {len(train_ds)}, val videos: {len(val_ds)}, test videos: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = LearnableConditioner(
        feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS, context_window_seconds=CONTEXT_WINDOW_SECONDS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_fallen_f1 = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            starts = batch["start_times"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(feats, starts, mask)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        y_true_val, y_pred_val = evaluate(model, val_loader, device)
        m_val = compute_all_metrics(y_true_val, y_pred_val, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
        val_fallen_f1 = m_val["fallen"]["f1"]
        print(f"[Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f} val_fallen_f1={val_fallen_f1*100:.2f}%")

        if val_fallen_f1 > best_val_fallen_f1:
            best_val_fallen_f1 = val_fallen_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val Fallen-F1: {best_val_fallen_f1*100:.2f}% -- loading best checkpoint for test eval.")
    model.load_state_dict(best_state)

    y_true_test, y_pred_test = evaluate(model, test_loader, device)
    m_test = compute_all_metrics(y_true_test, y_pred_test, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
    print("\n=== LEARNABLE CONDITIONER — Kết quả trên OF-ItW (OOPS) test ===")
    print_metrics_table(m_test)

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(best_state, ckpt_dir / "learnable_conditioner_best.pt")
    print(f"\nĐã lưu checkpoint: {ckpt_dir / 'learnable_conditioner_best.pt'}")


if __name__ == "__main__":
    main()
