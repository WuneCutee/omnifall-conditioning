"""Bước 4 — Nghiên cứu cải thiện KHÔNG huấn luyện lại backbone, theo từng tầng:

  A. Đánh giá baseline: ngẫu nhiên (như code gốc, 10 lần rút) vs crop giữa tất định vs TTA
     x 2 cách lấy mẫu frame (legacy / timefix).
  B. Hiệu chỉnh lệch phân bố lớp (label shift): EM không nhãn, hoặc prior từ nhãn ItW-train.
  C. Transition-Conditioning dựa luật: lưới (tau, W).
  D. Lọc tiến HMM với ma trận chuyển tiếp ước lượng từ nhãn train (Staged + Syn).
  E. Hybrid v8 (checkpoint có sẵn, 3 seed) trên đầu vào mới.

Quy tắc: MỌI lựa chọn đều theo điểm trên ItW-val = (Fallen-F1 + Fall∪Fallen-F1)/2;
test chỉ được tính cho cấu hình đã chọn (và các dòng tham chiếu cố định).

Usage:
    python pipeline/s04_posthoc_study.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from tcfall.config import CKPT_DIR, CLASS_NAMES, NUM_CLASSES, OUTPUTS, ROOT
from tcfall.conditioners import em_target_prior, estimate_transitions, load_hybrid
from tcfall.data import clean_labels, load_labels, load_split, transition_counts
from tcfall.metrics import core_metrics
from tcfall.study import apply_pipeline, flat, load_logits, load_with_spatial, source_prior, video_probs

OUT = OUTPUTS / "study"
SEEDS = (42, 43, 44)


def score(m: dict) -> float:
    return 0.5 * (m["fallen_f1"] + m["union_f1"])


def evaluate(items, cfg, hybrid=None) -> dict:
    return core_metrics(*flat(apply_pipeline(items, cfg, hybrid)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {(v, s): load_logits(v, s) for v in ("legacy", "timefix") for s in ("train", "val", "test")}
    has_spatial = (OUTPUTS / "logits" / "timefix_end_test.pkl").exists()
    if has_spatial:  # "spatial" = timefix + crop không gian trái/giữa/phải
        data.update({("spatial", s): load_with_spatial(s) for s in ("train", "val", "test")})
    rows = []

    def record(stage, name, cfg, splits=("val", "test"), hybrid=None, variant=None):
        v = variant or cfg["variant"]
        r = {"stage": stage, "name": name}
        for s in splits:
            m = evaluate(data[(v, s)], cfg, hybrid)
            r.update({f"{s}_{k}": round(val, 2) for k, val in m.items()})
            r[f"{s}_score"] = round(score(m), 2)
        rows.append(r)
        return r

    # ---------------- A. baseline: random vs center vs TTA, legacy vs timefix
    for v in ("legacy", "timefix"):
        draws = []
        for d in range(10):
            mv = evaluate(data[(v, "val")], {"variant": v, "random_draw": d})
            mt = evaluate(data[(v, "test")], {"variant": v, "random_draw": d})
            draws.append({**{f"val_{k}": x for k, x in mv.items()}, **{f"test_{k}": x for k, x in mt.items()}})
        dd = pd.DataFrame(draws)
        r = {"stage": "A", "name": f"{v} | ngẫu nhiên (10 lần rút, mean)"}
        r.update({k: round(dd[k].mean(), 2) for k in dd}); r.update({f"{k}_std": round(dd[k].std(), 2) for k in dd})
        rows.append(r)
        for crops in ("center", "tta3", "tta5", "tta9"):
            record("A", f"{v} | {crops}", {"variant": v, "crops": crops})
    if has_spatial:
        for crops in ("center", "tta3", "tta5", "tta9"):
            record("A", f"spatial | {crops}", {"variant": "spatial", "crops": crops})
    stageA = [r for r in rows if r["stage"] == "A" and "ngẫu nhiên" not in r["name"]]
    bestA = max(stageA, key=lambda r: r["val_score"])
    variant, crops = [x.strip() for x in bestA["name"].split("|")]
    base = {"variant": variant, "crops": crops}
    print(f"[A] chọn đầu vào baseline: {variant} + {crops}  (val score {bestA['val_score']})")

    # ---------------- B. label shift
    src = source_prior()
    oops, _ = clean_labels(load_labels("OOPS"))
    sp = load_split("OOPS")
    ytr = oops[oops.path.map(sp) == "train"]["label"].to_numpy()
    itw_prior = (np.bincount(ytr, minlength=NUM_CLASSES) + 1.0); itw_prior /= itw_prior.sum()
    em = {}
    for s in ("val", "test"):  # EM chạy trên xác suất KHÔNG NHÃN của chính tập đó (transductive)
        P = np.concatenate([video_probs(it, crops) for it in data[(variant, s)].values()])
        em[s] = em_target_prior(P, src)
    candB = [("không hiệu chỉnh", {**base})]
    for a in (0.25, 0.5, 0.75, 1.0):
        candB.append((f"prior ItW-train α={a}", {**base, "prior_alpha": a, "src_prior": src, "tgt_prior": itw_prior}))
    resB = []
    for name, cfg in candB:
        resB.append((evaluate(data[(variant, "val")], cfg), name, cfg))
    # EM: prior mục tiêu khác nhau giữa val/test nên đánh giá riêng
    for a in (0.25, 0.5, 0.75, 1.0):
        cfg_v = {**base, "prior_alpha": a, "src_prior": src, "tgt_prior": em["val"]}
        resB.append((evaluate(data[(variant, "val")], cfg_v), f"prior EM-không-nhãn α={a}", cfg_v))
    for m, name, cfg in resB:
        r = {"stage": "B", "name": name, **{f"val_{k}": round(x, 2) for k, x in m.items()}, "val_score": round(score(m), 2)}
        cfg_t = {**cfg, "tgt_prior": em["test"]} if "EM" in name else cfg
        mt = evaluate(data[(variant, "test")], cfg_t)
        r.update({f"test_{k}": round(x, 2) for k, x in mt.items()}); r["test_score"] = round(score(mt), 2)
        rows.append(r)
    bestB = max(resB, key=lambda x: score(x[0]))
    base_b = bestB[2]
    print(f"[B] chọn: {bestB[1]} (val score {score(bestB[0]):.2f})")

    def eff_prior(cfg):
        """Prior hiệu dụng sau hiệu chỉnh label shift — HMM chia xác suất cho prior này để ra likelihood."""
        if cfg.get("prior_alpha", 0) > 0:
            q = src * (cfg["tgt_prior"] / src) ** cfg["prior_alpha"]
            return q / q.sum()
        return src

    def for_test(cfg):  # EM: dùng prior ước lượng trên chính tập test (không nhãn)
        if cfg.get("prior_alpha", 0) > 0 and "EM" in bestB[1]:
            cfg = {**cfg, "tgt_prior": em["test"]}
            if cfg.get("cond") == "hmm":
                cfg["hmm_prior"] = eff_prior(cfg)
        return cfg

    # ---------------- C. rule grid
    grid = []
    for tau, W in itertools.product((0.5, 1, 2, 3, 4, 6, 8), (2, 3, 5, 7.5, 10, 15)):
        cfg = {**base_b, "cond": "rule", "tau": tau, "window": W}
        grid.append({"tau": tau, "W": W, **{k: round(x, 2) for k, x in evaluate(data[(variant, "val")], cfg).items()}})
    gdf = pd.DataFrame(grid); gdf["score"] = 0.5 * (gdf.fallen_f1 + gdf.union_f1)
    gdf.to_csv(OUT / "rule_grid_val.csv", index=False)
    bc = gdf.loc[gdf.score.idxmax()]
    tau, W = float(bc.tau), float(bc.W)
    print(f"[C] rule tốt nhất trên val: tau={tau}, W={W}")
    for name, t_, w_ in (("rule τ=4,W=10 (bài báo)", 4.0, 10.0), (f"rule τ={tau:g},W={W:g} (chọn trên val)", tau, W)):
        cfg = {**base_b, "cond": "rule", "tau": t_, "window": w_}
        r = {"stage": "C", "name": name}
        for s in ("val", "test"):
            m = evaluate(data[(variant, s)], cfg if s == "val" else for_test(cfg))
            r.update({f"{s}_{k}": round(x, 2) for k, x in m.items()}); r[f"{s}_score"] = round(score(m), 2)
        rows.append(r)

    # ---------------- D. HMM
    staged = pd.concat([clean_labels(load_labels(d))[0] for d in ("of-syn", "gmdcsa24", "le2i", "caucafall")])
    sp_all = {}
    for d in ("of-syn", "gmdcsa24", "le2i", "caucafall"):
        sp_all.update(load_split(d))
    counts = transition_counts(staged[staged.path.map(sp_all) == "train"])
    hmm_prior = eff_prior(base_b)
    hg = []
    for sl, stg in itertools.product((0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8), (0.15, 0.25, 0.35, 0.5, 0.75, 1.0)):
        A = estimate_transitions(counts, sl)
        cfg = {**base_b, "cond": "hmm", "A": A, "hmm_prior": hmm_prior, "strength": stg}
        m = evaluate(data[(variant, "val")], cfg)
        hg.append({"self_loop": sl, "strength": stg, **{k: round(x, 2) for k, x in m.items()}, "score": score(m)})
    hdf = pd.DataFrame(hg); hdf.to_csv(OUT / "hmm_grid_val.csv", index=False)
    bh = hdf.loc[hdf.score.idxmax()]
    A_best = estimate_transitions(counts, float(bh.self_loop))
    cfg_h = {**base_b, "cond": "hmm", "A": A_best, "hmm_prior": hmm_prior, "strength": float(bh.strength)}
    r = {"stage": "D", "name": f"HMM lọc tiến (self={bh.self_loop:g}, s={bh.strength:g})"}
    for s in ("val", "test"):
        m = evaluate(data[(variant, s)], cfg_h if s == "val" else for_test(cfg_h))
        r.update({f"{s}_{k}": round(x, 2) for k, x in m.items()}); r[f"{s}_score"] = round(score(m), 2)
    rows.append(r)
    print(f"[D] HMM tốt nhất: self_loop={bh.self_loop}, strength={bh.strength}")

    # ---------------- E. Hybrid v8 (checkpoint cũ) trên đầu vào mới, rule τ/W của bài báo (như lúc train)
    for seed in SEEDS:
        hyb = load_hybrid(CKPT_DIR / f"learnable_conditioner_v8_stsyn_kl0.3_seed{seed}_best.pt")
        for name, cfg0 in (("paper-input", {"variant": "legacy", "crops": "center"}), ("new-input", base_b)):
            cfg = {**cfg0, "cond": "hybrid", "tau": 4.0, "window": 10.0}
            r = {"stage": "E", "name": f"Hybrid v8 seed{seed} | {name}"}
            for s in ("val", "test"):
                v = cfg0.get("variant", variant)
                m = evaluate(data[(v, s)], cfg if s == "val" else for_test(cfg), hyb)
                r.update({f"{s}_{k}": round(x, 2) for k, x in m.items()}); r[f"{s}_score"] = round(score(m), 2)
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "posthoc_study.csv", index=False)
    cols = ["stage", "name", "val_score", "val_fallen_f1", "val_fall_f1", "val_union_f1",
            "test_fallen_f1", "test_fall_f1", "test_union_f1", "test_bacc"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(df[[c for c in cols if c in df]].to_string(index=False))

    best = {
        "variant": variant, "crops": crops, "prior": bestB[1],
        "prior_alpha": base_b.get("prior_alpha", 0.0),
        "itw_prior": itw_prior.tolist(), "src_prior": src.tolist(),
        "tau": tau, "window": W,
        "hmm_self_loop": float(bh.self_loop), "hmm_strength": float(bh.strength),
        "hmm_A": A_best.tolist(), "hmm_prior": hmm_prior.tolist(),
        "segment_hop": 1.0,
        # chế độ mặc định của app = conditioner không cần huấn luyện có điểm val cao hơn
        "app_mode": "hmm" if float(bh.score) > float(bc.score) else "rule",
    }
    (OUTPUTS / "best_config.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu {OUT/'posthoc_study.csv'} và {OUTPUTS/'best_config.json'}")


if __name__ == "__main__":
    main()
