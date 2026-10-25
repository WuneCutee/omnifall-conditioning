"""
Trích feature VideoMAE-Kinetics cho OF-Syn, đọc video TRỰC TIẾP từ tar (không giải nén
toàn bộ 12,000 video ra đĩa — chỉ đủ chỗ cho 1 video tại 1 thời điểm), lưu ra .h5 đúng
format mà omnifall-experiments (repo chính thức) cần:
  {OMNIFALL_ROOT}/of-syn/features/videomae-kinetics-7d5fps-1f/{path}.h5  (key "features", shape (T, 768))

Usage:
    python -m scripts.extract_of_syn_features --limit 50   # smoke test
    python -m scripts.extract_of_syn_features              # full 12,000 video
"""
from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from src.features import VideoMAEFeatureExtractor

TAR_PATH = Path("data/omnifall/data_files/omnifall-synthetic_av1.tar")
LABELS_CSV = Path("data/omnifall/labels/of-syn.csv")
OUTPUT_ROOT = Path("data/omnifall/of-syn/features/videomae-kinetics-7d5fps-1f")
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
FPS = 7.5
NUM_FRAMES = 16
STRIDE_SECONDS = 2 / 15


def load_wanted_paths() -> set[str]:
    """Toàn bộ path (không đuôi) cần feature — theo labels/of-syn.csv (mọi segment)."""
    df = pd.read_csv(LABELS_CSV, usecols=["path"])
    return set(df["path"].unique())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N video đầu (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wanted = load_wanted_paths()
    print(f"Cần feature cho {len(wanted)} video of-syn.")

    extractor = VideoMAEFeatureExtractor(
        model_name=MODEL_NAME, fps=FPS, num_frames=NUM_FRAMES, stride_seconds=STRIDE_SECONDS,
    )

    processed = 0
    skipped_existing = 0
    errors = []

    with tarfile.open(TAR_PATH) as tar:
        for member in tqdm(tar, desc="Streaming tar"):
            if not member.isfile() or not member.name.endswith(".mp4"):
                continue
            rel = member.name[2:] if member.name.startswith("./") else member.name
            path_no_ext = rel[:-4]
            if path_no_ext not in wanted:
                continue

            out_path = OUTPUT_ROOT / f"{path_no_ext}.h5"
            if out_path.exists() and not args.overwrite:
                skipped_existing += 1
                if args.limit is not None and processed >= args.limit:
                    break
                continue

            fileobj = tar.extractfile(member)
            if fileobj is None:
                errors.append((path_no_ext, "extractfile returned None"))
                continue

            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(fileobj.read())
                feat = extractor.extract(tmp_path)  # (T, 768)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(out_path, "w") as hf:
                    hf.create_dataset("features", data=feat)
            except Exception as e:  # noqa: BLE001
                errors.append((path_no_ext, str(e)))
            finally:
                os.remove(tmp_path)

            processed += 1
            if args.limit is not None and processed >= args.limit:
                break

    print(f"\nXong. Đã xử lý mới: {processed}, đã có sẵn (bỏ qua): {skipped_existing}, lỗi: {len(errors)}")
    for path_no_ext, err in errors[:20]:
        print(f"  [LỖI] {path_no_ext}: {err}")


if __name__ == "__main__":
    main()
