"""Bước 3 — chạy baseline trên feature OOPS mới, lưu logits theo từng video/segment.

Với mỗi segment lưu:
  crops  (9, 10): logits tại 9 vị trí cố định u = 0, 1/8, ..., 1  (crop giữa = chỉ số 4)
  random (10, 10): logits tại 10 vị trí ngẫu nhiên (seed cố định) = mô phỏng cách đánh giá
                   ngẫu nhiên của code gốc, để đo độ dao động của con số baseline.
Xuất ra outputs_v2/logits/{variant}_{split}.pkl

Usage:
    python pipeline/s03_cache_logits.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from tcfall.baseline import load_baseline, predict_logits
from tcfall.config import FEATURE_VARIANTS, OUTPUTS, feature_dir
from tcfall.data import clean_labels, load_labels, load_split
from tcfall.segments import crop_positions, gather, window_start

N_CROPS, N_RANDOM = 9, 10


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(FEATURE_VARIANTS),
                    help="vd: legacy,timefix  hoặc  timefix_start,timefix_end (TTA không gian)")
    variants = ap.parse_args().variants.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_baseline(device)
    labels, _ = clean_labels(load_labels("OOPS"))
    split = load_split("OOPS")
    labels["split"] = labels["path"].map(split)
    out_dir = OUTPUTS / "logits"; out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rand_u = {}  # cùng vị trí ngẫu nhiên cho cả 2 biến thể -> so sánh cặp

    for variant in variants:
        for sp in ("train", "val", "test"):
            result = {}
            for path, g in tqdm(labels[labels.split == sp].groupby("path", sort=True), desc=f"{variant}/{sp}"):
                g = g.sort_values("start")
                feats = np.load(feature_dir("OOPS", variant) / f"{Path(path).name}.npy").astype(np.float32)
                seqs, n = [], len(feats)
                for i, (s, e) in enumerate(zip(g.start, g.end)):
                    key = (path, i)
                    if key not in rand_u:
                        rand_u[key] = rng.random(N_RANDOM)
                    for u in list(crop_positions(N_CROPS)) + list(rand_u[key]):
                        seqs.append(gather(feats, window_start(s, e, u, n)))
                logits = predict_logits(model, torch.from_numpy(np.stack(seqs))).numpy()
                logits = logits.reshape(len(g), N_CROPS + N_RANDOM, -1)
                result[path] = {
                    "crops": logits[:, :N_CROPS], "random": logits[:, N_CROPS:],
                    "labels": g.label.to_numpy(), "starts": g.start.to_numpy(), "ends": g.end.to_numpy(),
                }
            with open(out_dir / f"{variant}_{sp}.pkl", "wb") as f:
                pickle.dump(result, f)
            print(f"{variant}/{sp}: {len(result)} video, {sum(len(v['labels']) for v in result.values())} segment")


if __name__ == "__main__":
    main()
