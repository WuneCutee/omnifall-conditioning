"""
Làm sạch file nhãn (không sửa file gốc, xuất ra data/omnifall/labels_clean/).

Các bước lọc, theo thứ tự:
  1. Bỏ dòng thiếu path/label/start/end
  2. Bỏ nhãn ngoài 0-9 (10 lớp lõi)
  3. Bỏ segment có end <= start (thời gian không hợp lệ)
  4. Bỏ segment trùng (cùng path, start, end)
  5. Bỏ segment ngắn hơn --min_dur giây (mặc định 0.5s)

Usage:
    python -m scripts.clean_labels --min_dur 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "omnifall" / "labels" / "dataset"
DST = ROOT / "data" / "omnifall" / "labels_clean"
DATASETS = ["of-syn", "GMDCSA24", "le2i", "caucafall", "OOPS"]


def clean(df: pd.DataFrame, min_dur: float) -> tuple[pd.DataFrame, dict]:
    report = {"rows_in": len(df)}

    n = len(df)
    df = df.dropna(subset=["path", "label", "start", "end"])
    report["null"] = n - len(df)

    n = len(df)
    df = df[(df["label"] >= 0) & (df["label"] <= 9)]
    report["bad_label"] = n - len(df)

    n = len(df)
    df = df[df["end"] > df["start"]]
    report["end_le_start"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates(subset=["path", "start", "end"])
    report["duplicate"] = n - len(df)

    n = len(df)
    df = df[(df["end"] - df["start"]) >= min_dur]
    report["too_short"] = n - len(df)

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_dur", type=float, default=0.5)
    args = parser.parse_args()

    DST.mkdir(parents=True, exist_ok=True)
    print(f"min_dur = {args.min_dur}s\n")
    print(f"{'dataset':10s} {'in':>7s} {'null':>5s} {'lbl':>5s} {'e<=s':>5s} {'dup':>5s} {'short':>6s} {'out':>7s}")
    for name in DATASETS:
        df = pd.read_csv(SRC / f"{name}.csv")
        out, r = clean(df, args.min_dur)
        out.to_csv(DST / f"{name}.csv", index=False)
        print(f"{name:10s} {r['rows_in']:7d} {r['null']:5d} {r['bad_label']:5d} {r['end_le_start']:5d} "
              f"{r['duplicate']:5d} {r['too_short']:6d} {r['rows_out']:7d}")
    print(f"\nĐã lưu: {DST}")


if __name__ == "__main__":
    main()
