"""Bước 2b — feature cho TTA không gian: với cách lấy mẫu timefix, trích thêm crop "start"
(trái/trên) và "end" (phải/dưới) bên cạnh crop giữa đã có ở bước 2.
Lưu vào features_v2/timefix_start, features_v2/timefix_end.

Usage:
    python pipeline/s02b_extract_spatial_crops.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from tcfall.config import LABELS_RAW, OOPS_VIDEO, feature_dir
from tcfall.features import VideoMAEExtractor
from tcfall.video import read_sampled


def main():
    paths = sorted(pd.read_csv(LABELS_RAW / "OOPS.csv", usecols=["path"])["path"].unique())
    ext = VideoMAEExtractor()
    for c in ("start", "end"):
        feature_dir("OOPS", f"timefix_{c}").mkdir(parents=True, exist_ok=True)
    for p in tqdm(paths, desc="spatial crops"):
        outs = {c: feature_dir("OOPS", f"timefix_{c}") / f"{Path(p).name}.npy" for c in ("start", "end")}
        if all(o.exists() for o in outs.values()):
            continue
        frames = read_sampled(str(OOPS_VIDEO / f"{p}.mp4"), ("timefix",))[0]["timefix"]
        for c, o in outs.items():
            np.save(o, ext.embed_windows(ext.preprocess(frames, crop=c)).astype(np.float16))
    print("Xong.")


if __name__ == "__main__":
    main()
