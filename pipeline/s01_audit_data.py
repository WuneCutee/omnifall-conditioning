"""Bước 1 — Kiểm toán dữ liệu (data audit) cho 5 nguồn dùng trong dự án:
OF-Syn, GMDCSA24, Le2i, CAUCAFall (staged) và OF-ItW/OOPS (wild test).

Sinh ra trong outputs_v2/data_audit/:
  overview.csv, cleaning.csv, class_distribution.csv, split_integrity.csv,
  temporal_integrity.csv, transition_<domain>.csv, fallen_context.csv, syn_metadata.csv
  + các hình fig_*.png dùng cho báo cáo.

Usage:
    python pipeline/s01_audit_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcfall.config import CLASS_NAMES, FALL, FALLEN, LIE_DOWN, LYING, NUM_CLASSES, OUTPUTS
from tcfall.data import clean_labels, load_labels, load_split, transition_counts

OUT = OUTPUTS / "data_audit"
DATASETS = ["of-syn", "gmdcsa24", "le2i", "caucafall", "OOPS"]
DOMAIN = {"of-syn": "Syn", "gmdcsa24": "Staged", "le2i": "Staged", "caucafall": "Staged", "OOPS": "ItW"}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9})


def overview(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for d, df in raw.items():
        rows.append({
            "dataset": d, "domain": DOMAIN[d], "videos": df["path"].nunique(), "segments": len(df),
            "hours": df["dur"].sum() / 3600, "seg_dur_mean": df["dur"].mean(),
            "seg_dur_median": df["dur"].median(), "seg_dur_max": df["dur"].max(),
            "segs_per_video": len(df) / df["path"].nunique(),
            "classes_present": int(df.loc[df.label < NUM_CLASSES, "label"].nunique()),
            "extended_label_segments": int((df["label"] >= NUM_CLASSES).sum()),
        })
    return pd.DataFrame(rows)


def split_integrity(raw) -> pd.DataFrame:
    rows = []
    for d, df in raw.items():
        sp = load_split(d)
        labeled = set(df["path"])
        in_split = set(sp)
        counts = pd.Series(list(sp.values())).value_counts()
        rows.append({
            "dataset": d, "train": int(counts.get("train", 0)), "val": int(counts.get("val", 0)),
            "test": int(counts.get("test", 0)), "conflict(path in >1 split)": int(counts.get("CONFLICT", 0)),
            "labeled_not_in_any_split": len(labeled - in_split),
            "split_entries_without_labels": len(in_split - labeled),
            "labeled_train": len([p for p in labeled if sp.get(p) == "train"]),
            "labeled_val": len([p for p in labeled if sp.get(p) == "val"]),
            "labeled_test": len([p for p in labeled if sp.get(p) == "test"]),
        })
        if d != "OOPS" and "subject" in df and (df["subject"] >= 0).any():
            subj = {s: set(df.loc[df.path.isin([p for p, v in sp.items() if v == s]), "subject"]) for s in ("train", "val", "test")}
            rows[-1]["subject_overlap_train_test"] = len(subj["train"] & subj["test"])
            rows[-1]["subject_overlap_train_val"] = len(subj["train"] & subj["val"])
    return pd.DataFrame(rows)


def temporal_integrity(raw) -> pd.DataFrame:
    rows = []
    for d, df in raw.items():
        df = df.sort_values(["path", "start"])
        nxt_start = df.groupby("path")["start"].shift(-1)
        gap = nxt_start - df["end"]
        rows.append({
            "dataset": d,
            "overlap_pairs(>10ms)": int((gap < -0.01).sum()),
            "gap_pairs(>100ms)": int((gap > 0.1).sum()),
            "first_seg_not_at_0": int((df.groupby("path")["start"].min() > 0.1).sum()),
            "segments_<0.5s": int((df["dur"] < 0.5).sum()),
            "segments_<0.25s": int((df["dur"] < 0.25).sum()),
            "segments_>10s": int((df["dur"] > 10).sum()),
        })
    return pd.DataFrame(rows)


def fallen_context(df: pd.DataFrame, dataset_label: str) -> dict:
    """Với mỗi segment fallen/lying: segment ngay trước là gì, và cách hành động chuyển tiếp
    tương ứng (fall/lie_down) gần nhất bao lâu — căn cứ dữ liệu cho cửa sổ W và chu kỳ bán rã τ."""
    res = {"domain": dataset_label}
    for static, trans, name in ((FALLEN, FALL, "fallen"), (LYING, LIE_DOWN, "lying")):
        prev_is_trans, delays, n = 0, [], 0
        for _, g in df.sort_values(["path", "start"]).groupby("path", sort=False):
            lab, st, en = g["label"].to_numpy(), g["start"].to_numpy(), g["end"].to_numpy()
            for i in np.where(lab == static)[0]:
                n += 1
                if i > 0 and lab[i - 1] == trans:
                    prev_is_trans += 1
                prior = np.where((lab[:i] == trans))[0]
                if len(prior):
                    delays.append(st[i] - st[prior[-1]])  # từ lúc BẮT ĐẦU transient gần nhất
        delays = np.array(delays)
        res.update({
            f"{name}_segments": n,
            f"{name}_prev_is_{CLASS_NAMES[trans]}_%": 100 * prev_is_trans / max(n, 1),
            f"{name}_has_{CLASS_NAMES[trans]}_before_%": 100 * len(delays) / max(n, 1),
            f"{name}_delay_median_s": float(np.median(delays)) if len(delays) else np.nan,
            f"{name}_delay_p90_s": float(np.percentile(delays, 90)) if len(delays) else np.nan,
            f"{name}_within_10s_%": 100 * float((delays <= 10).mean()) if len(delays) else np.nan,
        })
    return res


def class_distribution(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    groups = {}
    staged = pd.concat([clean[d] for d in ("gmdcsa24", "le2i", "caucafall")])
    groups["Staged (3 bộ)"] = staged
    groups["OF-Syn"] = clean["of-syn"]
    oops = clean["OOPS"]; sp = load_split("OOPS")
    for s in ("train", "val", "test"):
        groups[f"ItW {s}"] = oops[oops.path.map(sp) == s]
    rows = []
    for g, df in groups.items():
        c = df["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        rows.append({"group": g, "segments": int(c.sum()), **{CLASS_NAMES[k]: int(c[k]) for k in range(NUM_CLASSES)}})
    return pd.DataFrame(rows)


def fig_class_distribution(cd: pd.DataFrame):
    props = cd.set_index("group")[CLASS_NAMES]
    props = props.div(props.sum(1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(NUM_CLASSES); w = 0.8 / len(props)
    for i, (g, r) in enumerate(props.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, r.values, w, label=g)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("% số segment"); ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_class_distribution.png"); plt.close(fig)


def fig_durations(clean):
    allc = pd.concat([df.assign(domain=DOMAIN[d]) for d, df in clean.items()])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    data, labels, pos = [], [], []
    colors = {"Syn": "#4a90d9", "Staged": "#9e9e9e", "ItW": "#e07b39"}
    for k in range(NUM_CLASSES):
        for j, dom in enumerate(("Staged", "Syn", "ItW")):
            v = allc[(allc.label == k) & (allc.domain == dom)]["dur"].clip(upper=20).to_numpy()
            data.append(v if len(v) else np.array([np.nan])); pos.append(k * 4 + j)
    bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(list(colors.values())[[1, 0, 2][i % 3]])
    ax.set_xticks([k * 4 + 1 for k in range(NUM_CLASSES)]); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Độ dài segment (s)")
    for dom, c in colors.items():
        ax.plot([], [], color=c, lw=6, label=dom)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_segment_durations.png"); plt.close(fig)


def fig_transitions(mats: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 4))
    for ax, (name, M) in zip(np.atleast_1d(axes), mats.items()):
        P = M / M.sum(1, keepdims=True).clip(min=1)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        ax.set_title(f"{name}: P(nhãn kế tiếp | nhãn hiện tại)", fontsize=8)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if P[i, j] >= 0.15:
                    ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if P[i, j] > .5 else "black")
    fig.tight_layout(); fig.savefig(OUT / "fig_transition_matrices.png"); plt.close(fig)


def syn_metadata(df: pd.DataFrame) -> pd.DataFrame:
    vids = df.drop_duplicates("path")
    rows = []
    for col in ("age_group", "gender_presentation", "race_ethnicity_omb", "bmi_band",
                "environment_category", "camera_shot", "speed", "camera_elevation", "camera_distance"):
        for k, v in vids[col].value_counts().items():
            rows.append({"attribute": col, "value": k, "videos": int(v), "%": 100 * v / len(vids)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {d: load_labels(d) for d in DATASETS}
    ov = overview(raw); ov.to_csv(OUT / "overview.csv", index=False)

    clean, rep_rows = {}, []
    for d, df in raw.items():
        c, rep = clean_labels(df)
        clean[d] = c
        rep_rows.append({"dataset": d, "rows_in": len(df), **{k: v for k, v in rep.steps}, "rows_out": len(c)})
    pd.DataFrame(rep_rows).to_csv(OUT / "cleaning.csv", index=False)

    split_integrity(raw).to_csv(OUT / "split_integrity.csv", index=False)
    temporal_integrity(raw).to_csv(OUT / "temporal_integrity.csv", index=False)

    cd = class_distribution(clean); cd.to_csv(OUT / "class_distribution.csv", index=False)
    fig_class_distribution(cd); fig_durations(clean)

    staged = pd.concat([clean[d] for d in ("gmdcsa24", "le2i", "caucafall")])
    mats = {"Staged": transition_counts(staged), "OF-Syn": transition_counts(clean["of-syn"]),
            "OF-ItW": transition_counts(clean["OOPS"])}
    for k, M in mats.items():
        pd.DataFrame(M, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT / f"transition_{k}.csv")
    fig_transitions(mats)

    fc = pd.DataFrame([fallen_context(staged, "Staged"), fallen_context(clean["of-syn"], "OF-Syn"),
                       fallen_context(clean["OOPS"], "OF-ItW")])
    fc.to_csv(OUT / "fallen_context.csv", index=False)
    syn_metadata(raw["of-syn"]).to_csv(OUT / "syn_metadata.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        for f in ("overview", "cleaning", "split_integrity", "temporal_integrity", "class_distribution", "fallen_context"):
            print(f"\n=== {f} ===\n", pd.read_csv(OUT / f"{f}.csv").round(2).to_string(index=False))
    print(f"\nĐã lưu vào {OUT}")


if __name__ == "__main__":
    main()
