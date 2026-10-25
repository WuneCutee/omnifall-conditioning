"""
Trích feature VideoMAE-Kinetics cho OF-ItW (OOPS), đọc từ thư mục video thật đã chuẩn bị
bằng data/omnifall/prepare_oops_videos.py, lưu ra .h5 đúng format omnifall-experiments cần:
  {OMNIFALL_ROOT}/OOPS/features/videomae-kinetics-7d5fps-1f/{path}.h5

Usage:
    python -m scripts.extract_oops_features --limit 20   # smoke test
    python -m scripts.extract_oops_features              # toàn bộ 818 video
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from src.features import VideoMAEFeatureExtractor

VIDEO_ROOT = Path("data/omnifall/OOPS/video")
LABELS_CSV = Path("data/omnifall/labels/OOPS.csv")
OUTPUT_ROOT = Path("data/omnifall/OOPS/features/videomae-kinetics-7d5fps-1f")
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
FPS = 7.5
NUM_FRAMES = 16
STRIDE_SECONDS = 2 / 15


def load_wanted_paths() -> set[str]:
    df = pd.read_csv(LABELS_CSV, usecols=["path"])
    return set(df["path"].unique())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wanted = load_wanted_paths()
    print(f"Cần feature cho {len(wanted)} video OOPS.")

    extractor = VideoMAEFeatureExtractor(
        model_name=MODEL_NAME, fps=FPS, num_frames=NUM_FRAMES, stride_seconds=STRIDE_SECONDS,
    )

    processed = 0
    skipped_existing = 0
    missing = []
    errors = []

    for path_no_ext in tqdm(sorted(wanted)):
        out_path = OUTPUT_ROOT / f"{path_no_ext}.h5"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        video_path = VIDEO_ROOT / f"{path_no_ext}.mp4"
        if not video_path.exists():
            missing.append(path_no_ext)
            continue

        try:
            feat = extractor.extract(str(video_path))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("features", data=feat)
        except Exception as e:  # noqa: BLE001
            errors.append((path_no_ext, str(e)))

        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    print(f"\nXong. Đã xử lý mới: {processed}, đã có sẵn: {skipped_existing}, thiếu video: {len(missing)}, lỗi: {len(errors)}")
    for p in missing[:10]:
        print(f"  [THIẾU] {p}")
    for p, err in errors[:10]:
        print(f"  [LỖI] {p}: {err}")


if __name__ == "__main__":
    main()
