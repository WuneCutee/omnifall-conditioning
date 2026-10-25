"""Bước 8 — Tổng hợp kết quả cuối cho báo cáo (bảng + kiểm định + hình) vào outputs_v2/final/.

Hai thước đo, báo cáo tách biệt:
  (1) THEO SEGMENT — protocol OmniFall: ranh giới segment lấy từ nhãn (92% cặp liền kề khác lớp
      -> conditioner có thể khai thác thông tin oracle này).
  (2) STREAMING — mô phỏng camera, không dùng ranh giới nhãn; nhãn so ở t - 2.2 s (bù trễ cửa sổ
      nhân quả). Đây là thước đo chính cho ứng dụng realtime.

Usage:
    python pipeline/s08_final_report.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from tcfall.config import CKPT_DIR, FALL, FALLEN, LIE_DOWN, LYING, OUTPUTS
from tcfall.conditioners import load_hybrid
from tcfall.metrics import core_metrics, mcnemar, video_bootstrap
from tcfall.study import apply_pipeline, flat, load_logits

OUT = OUTPUTS / "final"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9})
K = ("fallen_f1", "fall_f1", "union_f1", "bacc")


def ms(vals):
    v = np.asarray(vals, float)
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v[0]:.1f}"


def segment_table():
    study = pd.read_csv(OUTPUTS / "study" / "posthoc_study.csv")
    rows = []

    def from_study(name, label):
        r = study[study.name == name].iloc[0]
        rows.append({"method": label, **{k: f"{r[f'test_{k}']:.1f}" for k in K},
                     **({f"{k}_sd": r.get(f"test_{k}_std") for k in K} if f"test_{K[0]}_std" in r else {})})

    r = study[study.name == "legacy | ngẫu nhiên (10 lần rút, mean)"].iloc[0]
    rows.append({"method": "Baseline St+Sy — đánh giá ngẫu nhiên như code gốc (10 lần rút)",
                 **{k: f"{r[f'test_{k}']:.1f} ± {r[f'test_{k}_std']:.1f}" for k in K}})
    from_study("legacy | center", "Baseline — crop giữa tất định")
    from_study("timefix | center", "Baseline — lấy mẫu timefix, crop giữa")
    from_study("spatial | center", "Baseline — timefix + TTA không gian (3 crop)")
    from_study("rule τ=4,W=10 (bài báo)", "Rule-conditioning τ=4,W=10 (bài báo) *")
    rr = study[(study.stage == "C") & ~study.name.str.contains("bài báo")].iloc[0]
    from_study(rr["name"], f"Rule-conditioning {rr['name'].split(' (')[0][5:]} (chọn trên val) *")
    hm = study[study.stage == "D"].iloc[0]
    from_study(hm["name"], "HMM lọc tiến, self-loop≈0 (chọn trên val) * †")
    hv8 = study[study.name.str.contains("Hybrid v8") & study.name.str.contains("paper-input")]
    rows.append({"method": "Hybrid v8 (bài báo, 3 seed) *", **{k: ms(hv8[f"test_{k}"]) for k in K}})
    abl = OUTPUTS / "hybrid" / "hybrid_grid_rule_paper.csv"
    if abl.exists():
        a = pd.read_csv(abl)
        for lam, g in a.groupby("lambda"):
            rows.append({"method": f"Hybrid huấn luyện lại, thiết lập bài báo, λ_KL={lam:g} ({len(g)} seed) *",
                         **{k: ms(g[f"test_{k}"]) for k in K}})
    ft = OUTPUTS / "finetune" / "finetune_results.csv"
    if ft.exists():
        f = pd.read_csv(ft)
        for ev, lab in (("center_tta1", "crop giữa"), ("spatial_tta5", "TTA không gian + thời gian")):
            g = f[f["eval"] == ev]
            rows.append({"method": f"Fine-tune classifier trên ItW-train ({lab}, 3 seed) ‡",
                         **{k: ms(g[f"test_{k}"]) for k in K}})
    df = pd.DataFrame(rows)[["method", *K]]
    df.columns = ["Phương pháp", "Fallen-F1", "Fall-F1", "Fall∪Fallen F1", "Balanced Acc"]
    return df


def stream_table():
    rows = []
    for tag, lab in (("stream", "St+Sy"), ("stream_ft", "St+Sy → fine-tune ItW-train ‡")):
        f = OUTPUTS / tag / "stream_test_step.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        hy = d[d.name.str.startswith("hybrid")]
        for _, r in d[~d.name.str.startswith("hybrid")].iterrows():
            rows.append({"baseline": lab, "conditioner": f"{r['name']} [{r['input']}]",
                         **{f"{k}@2.2s": f"{r[f'lag2.2_{k}']:.1f}" for k in K[:3]},
                         **{f"{k}@0s": f"{r[f'lag0.0_{k}']:.1f}" for k in K[:3]}})
        if len(hy):
            rows.append({"baseline": lab, "conditioner": f"Hybrid v8 (3 seed) [{hy.input.iloc[0]}]",
                         **{f"{k}@2.2s": ms(hy[f"lag2.2_{k}"]) for k in K[:3]},
                         **{f"{k}@0s": ms(hy[f"lag0.0_{k}"]) for k in K[:3]}})
    return pd.DataFrame(rows)


def event_table():
    rows = []
    for tag, lab in (("stream", "St+Sy"), ("stream_ft", "Fine-tune ItW ‡")):
        f = OUTPUTS / tag / "stream_test_events.csv"
        if f.exists():
            d = pd.read_csv(f)
            d.insert(0, "baseline", lab)
            rows.append(d)
    return pd.concat(rows) if rows else pd.DataFrame()


def significance():
    """Bootstrap theo video (572 video) + McNemar theo segment, cho các so sánh chính."""
    out = []
    test_legacy = load_logits("legacy", "test")
    base = apply_pipeline(test_legacy, {"crops": "center"})
    comps = {"Rule τ=4,W=10 (bài báo)": apply_pipeline(test_legacy, {"crops": "center", "cond": "rule", "tau": 4, "window": 10})}
    for s in (42, 43, 44):
        comps[f"Hybrid v8 seed{s}"] = apply_pipeline(test_legacy, {"crops": "center", "cond": "hybrid", "tau": 4, "window": 10},
                                                     load_hybrid(CKPT_DIR / f"learnable_conditioner_v8_stsyn_kl0.3_seed{s}_best.pt"))
    fp = OUTPUTS / "finetune" / "finetune_test_preds.pkl"
    if fp.exists():
        ft = pickle.load(open(fp, "rb"))
        for s in (42, 43, 44):
            comps[f"Fine-tune ItW seed{s} (crop giữa)"] = ft[f"seed{s}_center_tta1"]
    y = np.concatenate([a for a, _ in base])
    for name, pv in comps.items():
        assert all((a == b).all() for (a, _), (b, _) in zip(base, pv)), "thứ tự video lệch"
        r = {"so sánh (B vs baseline crop giữa)": name, "thước đo": "theo segment"}
        for m in ("fallen_f1", "fall_f1", "union_f1"):
            bt = video_bootstrap([(base[i][0], base[i][1], pv[i][1]) for i in range(len(base))], m)
            r[f"Δ{m}"] = round(bt["mean"], 2); r[f"{m} CI95"] = f"[{bt['lo']:+.1f}, {bt['hi']:+.1f}]"
        pa, pb = flat(base)[1], flat(pv)[1]
        tu = np.isin(y, [FALL, FALLEN])
        mc = mcnemar(np.isin(pa, [FALL, FALLEN]) == tu, np.isin(pb, [FALL, FALLEN]) == tu)
        r.update({"McNemar union: chỉ A đúng": mc["only_a"], "chỉ B đúng": mc["only_b"], "p": f"{mc['p']:.2g}"})
        out.append(r)
    # streaming: so sánh với baseline streaming cùng đầu vào, nhãn tại t-2.2 s
    for tag in ("stream", "stream_ft"):
        f = OUTPUTS / tag / "stream_test_preds.pkl"
        if not f.exists():
            continue
        d = pickle.load(open(f, "rb"))
        ref_key = next(k for k in d["preds"] if k.startswith("baseline | timefix"))
        for key, preds in d["preds"].items():
            if key == ref_key:
                continue
            inp = key.split(" | ")[1]
            per = []
            for path, (t, g) in d["streams"][inp].items():
                idx = np.searchsorted(g.start.to_numpy(), t - 2.2, side="right") - 1
                ok = (idx >= 0) & (t - 2.2 < g.end.to_numpy()[np.clip(idx, 0, None)])
                per.append((g.label.to_numpy()[idx[ok]], d["preds"][ref_key][path][ok], preds[path][ok]))
            r = {"so sánh (B vs baseline crop giữa)": f"[{tag}] {key}", "thước đo": "streaming @2.2s"}
            for m in ("fallen_f1", "fall_f1", "union_f1"):
                bt = video_bootstrap(per, m)
                r[f"Δ{m}"] = round(bt["mean"], 2); r[f"{m} CI95"] = f"[{bt['lo']:+.1f}, {bt['hi']:+.1f}]"
            out.append(r)
    return pd.DataFrame(out)


def fig_confusion():
    test_legacy = load_logits("legacy", "test")
    pairs = [("Baseline St+Sy (crop giữa)", apply_pipeline(test_legacy, {"crops": "center"}))]
    fp = OUTPUTS / "finetune" / "finetune_test_preds.pkl"
    if fp.exists():
        ft = pickle.load(open(fp, "rb"))
        pairs.append(("Fine-tune ItW-train (3 seed gộp)",
                      [x for s in (42, 43, 44) for x in ft[f"seed{s}_center_tta1"]]))
    hv8 = load_hybrid(CKPT_DIR / "learnable_conditioner_v8_stsyn_kl0.3_seed42_best.pt")
    pairs.insert(1, ("Hybrid v8 seed42 (bài báo)",
                     apply_pipeline(test_legacy, {"crops": "center", "cond": "hybrid", "tau": 4, "window": 10}, hv8)))
    idx, names = [FALL, LIE_DOWN, FALLEN, LYING], ["fall", "lie_down", "fallen", "lying"]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2))
    for ax, (title, pv) in zip(axes, pairs):
        y, p = flat(pv)
        cm = confusion_matrix(y, p, labels=range(10))
        M = np.column_stack([cm[np.ix_(idx, idx)], cm[idx].sum(1) - cm[np.ix_(idx, idx)].sum(1)]) / cm[idx].sum(1, keepdims=True)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(names + ["6 lớp khác"], rotation=30)
        ax.set_yticks(range(4)); ax.set_yticklabels([f"{n} (n={c})" for n, c in zip(names, cm[idx].sum(1))])
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f"{100*M[i,j]:.1f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] > .5 else "k")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật")
    fig.suptitle("Recall theo hàng (%) — mẫu số là TẤT CẢ segment của lớp (ItW-test)")
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion_full.png"); plt.close(fig)


def fig_bars(seg: pd.DataFrame, stream: pd.DataFrame):
    def num(s):
        return float(str(s).split(" ±")[0])

    def err(s):
        return float(str(s).split("± ")[1]) if "±" in str(s) else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, col in zip(axes, ("Fallen-F1", "Fall-F1", "Fall∪Fallen F1")):
        names = list(seg["Phương pháp"])
        v, e = [num(x) for x in seg[col]], [err(x) for x in seg[col]]
        colors = ["#2e7d32" if "Fine-tune" in n else "#e07b39" if "Hybrid" in n else "#4a90d9" if ("Rule" in n or "HMM" in n)
                  else "#9e9e9e" for n in names]
        ax.barh(range(len(v)), v, xerr=e, color=colors, edgecolor="k", lw=.4, capsize=2)
        ax.set_yticks(range(len(v))); ax.set_yticklabels(names if ax is axes[0] else [], fontsize=6.5)
        ax.invert_yaxis(); ax.set_title(f"{col} — theo segment (ItW-test)"); ax.grid(axis="x", alpha=.3)
        for i, x in enumerate(v):
            ax.text(x + .5, i, f"{x:.1f}", va="center", fontsize=6.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_segment_bars.png"); plt.close(fig)

    if len(stream):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        labels = [f"{b}: {c}" for b, c in zip(stream.baseline, stream.conditioner)]
        for ax, k, t in zip(axes, ("fallen_f1@2.2s", "fall_f1@2.2s", "union_f1@2.2s"), ("Fallen-F1", "Fall-F1", "Fall∪Fallen F1")):
            v, e = [num(x) for x in stream[k]], [err(x) for x in stream[k]]
            colors = ["#2e7d32" if "fine-tune" in b else "#9e9e9e" for b in stream.baseline]
            ax.barh(range(len(v)), v, xerr=e, color=colors, edgecolor="k", lw=.4, capsize=2)
            ax.set_yticks(range(len(v))); ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=6.5)
            ax.invert_yaxis(); ax.set_title(f"{t} — streaming, nhãn tại t−2.2 s"); ax.grid(axis="x", alpha=.3)
            for i, x in enumerate(v):
                ax.text(x + .5, i, f"{x:.1f}", va="center", fontsize=6.5)
        fig.tight_layout(); fig.savefig(OUT / "fig_stream_bars.png"); plt.close(fig)


def fig_tradeoff(ev: pd.DataFrame):
    if not len(ev):
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for (b, n, i), g in ev.groupby(["baseline", "name", "input"], sort=False):
        if n.startswith("hybrid") and "seed42" not in n:
            continue
        g = g.sort_values("alarm_on_s")
        ls = "-" if "Fine" in b else "--"
        ax.plot(g.false_alarms_per_hour, g["episode_recall_%"], ls, marker="o", ms=3, label=f"{b}: {n} [{i}]")
        for _, r in g.iterrows():
            ax.annotate(f"{r.alarm_on_s:g}", (r.false_alarms_per_hour, r["episode_recall_%"]), fontsize=5.5,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Báo động giả / giờ (ItW-test, clip 'fail' — khó hơn môi trường chăm sóc)")
    ax.set_ylabel("% đợt fallen được báo động")
    ax.set_title("Đánh đổi theo ngưỡng báo động (số trên điểm = giây fallen liên tục)")
    ax.grid(alpha=.3); ax.legend(fontsize=5.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_stream_tradeoff.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = segment_table(); seg.to_csv(OUT / "table_segment.csv", index=False)
    st = stream_table(); st.to_csv(OUT / "table_stream.csv", index=False)
    ev = event_table(); ev.to_csv(OUT / "table_events.csv", index=False)
    sig = significance(); sig.to_csv(OUT / "table_significance.csv", index=False)
    fig_confusion(); fig_bars(seg, st); fig_tradeoff(ev)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        for name, d in (("THEO SEGMENT", seg), ("STREAMING", st), ("KIỂM ĐỊNH", sig)):
            print(f"\n=== {name} ===\n{d.to_string(index=False)}")
    print("\n* dùng ranh giới segment từ nhãn  † khai thác 'nhãn đổi ở mỗi ranh giới' (oracle)  ‡ có dùng nhãn ItW-train")
    print(f"Đã lưu {OUT}")


if __name__ == "__main__":
    main()
