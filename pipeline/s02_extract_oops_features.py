"""Bước 2 — trích feature VideoMAE cho toàn bộ 818 video OF-ItW (OOPS), 2 biến thể lấy mẫu
(legacy, timefix) trong cùng 1 lượt giải mã. Lưu .npy float16 + bảng thông tin video.

Usage:
    python pipeline/s02_extract_oops_features.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from tcfall.config import FEATURE_VARIANTS, LABELS_RAW, OOPS_VIDEO, OUTPUTS, feature_dir
from tcfall.features import VideoMAEExtractor
from tcfall.video import read_sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(pd.read_csv(LABELS_RAW / "OOPS.csv", usecols=["path"])["path"].unique())
    if args.limit:
        paths = paths[:args.limit]
    for v in FEATURE_VARIANTS:
        feature_dir("OOPS", v).mkdir(parents=True, exist_ok=True)

    ext = VideoMAEExtractor()
    rows = []
    t0 = time.time()
    for p in tqdm(paths, desc="OOPS features"):
        outs = {v: feature_dir("OOPS", v) / f"{Path(p).name}.npy" for v in FEATURE_VARIANTS}
        video = OOPS_VIDEO / f"{p}.mp4"
        frames, info = read_sampled(str(video), FEATURE_VARIANTS)
        if not all(o.exists() for o in outs.values()):
            feats = ext.extract_variants(frames)
            for v, f in feats.items():
                np.save(outs[v], f.astype(np.float16))
        info.update({"path": p, **{f"n_{v}": len(frames[v]) for v in FEATURE_VARIANTS}})
        rows.append(info)

    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUTS / "oops_video_info.csv", index=False)
    print(f"Xong {len(rows)} video trong {(time.time()-t0)/60:.1f} phút -> {feature_dir('OOPS', 'timefix').parent}")


if __name__ == "__main__":
    main()
