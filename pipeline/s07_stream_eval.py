"""Bước 7 — Đánh giá STREAMING (không dùng ranh giới segment từ nhãn) — thước đo chính.

Vì sao: ở đánh giá theo segment, 92% cặp segment ItW liền kề có nhãn khác nhau (ranh giới
segment = nơi nhãn đổi), nên conditioner nào "biết" điều đó (HMM self-loop thấp, Hybrid học
từ chuỗi segment) được lợi từ thông tin oracle. Ở đây mô phỏng đúng camera:
  tại mỗi bước 0.133 s chỉ dùng 18 feature quá khứ -> baseline -> StreamConditioner
  (pseudo-segment mỗi `hop` s, conditioner nhân quả, báo động hysteresis) — giống hệt app.

Quy trình: tinh chỉnh mọi tham số trên luồng ItW-VAL, đánh giá 1 lần trên ItW-TEST.
  * metric theo bước: so dự đoán tại t với nhãn tại t - lag (lag 0 / 1.1 / 2.2 s; cửa sổ nhân
    quả 4.4 s nên tâm ngữ cảnh trễ ~2.2 s); điểm chọn = (Fallen-F1 + Fall∪Fallen-F1)/2 tại lag 2.2
  * metric sự kiện: % đợt fallen được báo động, độ trễ, báo động giả / giờ, theo ngưỡng báo động
  * HMM: ma trận chuyển tiếp đếm từ nhãn train (Staged+Syn) LẤY MẪU MỖI `hop` GIÂY

Ghi cấu hình streaming tốt nhất vào outputs_v2/best_config.json (khoá "stream") cho app.

Usage:
    python pipeline/s07_stream_eval.py
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tcfall.baseline import load_baseline, predict_logits
from tcfall.config import CKPT_DIR, FALLEN, NUM_FEATURES, OUTPUTS, TARGET_FPS, WINDOW_FRAMES, feature_dir
from tcfall.conditioners import load_hybrid, sharpen_self_loop, transitions_from_counts
from tcfall.data import clean_labels, load_labels, load_split, transition_counts_timesampled
from tcfall.metrics import core_metrics
from tcfall.realtime import EngineConfig, StreamConditioner
from tcfall.study import source_prior

import os
TAG = os.environ.get("TCFALL_STREAM_TAG", "")
OUT = OUTPUTS / ("stream" + (f"_{TAG}" if TAG else ""))
LAGS = (0.0, 1.1, 2.2)
SEL_LAG = 2.2
CROPS = {"timefix": ["timefix"], "spatial": ["timefix", "timefix_start", "timefix_end"]}


def episodes(g: pd.DataFrame) -> list[tuple[float, float]]:
    eps = []
    for s, e in zip(g[g.label == FALLEN].start, g[g.label == FALLEN].end):
        if eps and s - eps[-1][1] < 0.5:
            eps[-1] = (eps[-1][0], e)
        else:
            eps.append((s, e))
    return eps


def build_streams(model, labels, split, inputs):
    """{input: {path: (t, P, g)}} — xác suất baseline nhân quả tại mỗi bước."""
    sp = load_split("OOPS")
    sub = labels[labels.path.map(sp) == split]
    need = sorted({c for i in inputs for c in CROPS[i]})
    out = {i: {} for i in inputs}
    for path, g in tqdm(sub.groupby("path"), desc=f"streams {split}"):
        name = f"{Path(path).name}.npy"
        probs = {}
        for v in need:
            f = np.load(feature_dir("OOPS", v) / name).astype(np.float32)
            if len(f) < NUM_FEATURES:
                break
            seqs = np.stack([f[i - NUM_FEATURES + 1:i + 1] for i in range(NUM_FEATURES - 1, len(f))])
            probs[v] = torch.softmax(predict_logits(model, torch.from_numpy(seqs)), -1).numpy()
            n = len(f)
        if len(probs) < len(need):
            continue
        t = (np.arange(NUM_FEATURES - 1, n) + WINDOW_FRAMES - 1) / TARGET_FPS
        for i in inputs:
            out[i][path] = (t, np.mean([probs[c] for c in CROPS[i]], 0), g.sort_values("start"))
    return out


def run_preds(streams, cfg: EngineConfig, hybrid=None) -> dict:
    preds = {}
    for path, (t, P, g) in streams.items():
        sc = StreamConditioner(cfg, hybrid)
        preds[path] = np.array([int(sc.update(tt, p).probs.argmax()) for tt, p in zip(t, P)])
    return preds


def gt_at(g, times):
    idx = np.searchsorted(g.start.to_numpy(), times, side="right") - 1
    ok = (idx >= 0) & (times < g.end.to_numpy()[np.clip(idx, 0, None)])
    return g.label.to_numpy()[np.clip(idx, 0, None)], ok


def step_metrics(streams, preds, lag) -> dict:
    ys, ps = [], []
    for path, (t, P, g) in streams.items():
        y, ok = gt_at(g, t - lag)
        ys.append(y[ok]); ps.append(preds[path][ok])
    return core_metrics(np.concatenate(ys), np.concatenate(ps))


def alarms_from(t, pred, on_s, off_s=3.0):
    """Báo động hysteresis (giống StreamConditioner._update_alarm) tính lại từ chuỗi dự đoán."""
    ev, alarm, since, clear = [], False, None, None
    for tt, k in zip(t, pred):
        if k == FALLEN:
            clear = None
            since = tt if since is None else since
            if not alarm and tt - since >= on_s:
                alarm = True; ev.append(tt)
        else:
            since = None
            if alarm:
                clear = tt if clear is None else clear
                if tt - clear >= off_s:
                    alarm = False
    return ev


def event_metrics(streams, preds, on_s) -> dict:
    det, lat, fa, n_eps, total_s = 0, [], 0, 0, 0.0
    for path, (t, P, g) in streams.items():
        ev, eps, used = alarms_from(t, preds[path], on_s), episodes(g), set()
        for s_, e_ in eps:
            n_eps += 1
            hit = [a for a in ev if s_ <= a <= e_ + 5.0]
            if hit:
                det += 1; lat.append(hit[0] - s_); used.update(hit)
        fa += sum(1 for a in ev if a not in used and not any(s_ - 1.0 <= a <= e_ + 5.0 for s_, e_ in eps))
        total_s += t[-1] - t[0] + 1 / TARGET_FPS
    return {"fallen_episodes": n_eps, "episode_recall_%": round(100 * det / max(n_eps, 1), 1),
            "latency_median_s": round(float(np.median(lat)), 2) if lat else np.nan,
            "false_alarms_per_hour": round(fa / (total_s / 3600), 1), "video_hours": round(total_s / 3600, 2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import os
    device = os.environ.get("TCFALL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_baseline(device)
    labels, _ = clean_labels(load_labels("OOPS"))
    inputs = ("timefix", "spatial")
    val = build_streams(model, labels, "val", inputs)
    test = build_streams(model, labels, "test", inputs)

    # ma trận chuyển tiếp lấy mẫu theo thời gian, từ nhãn TRAIN của Staged + Syn
    src = pd.concat([clean_labels(load_labels(d))[0] for d in ("of-syn", "gmdcsa24", "le2i", "caucafall")])
    sp = {}
    for d in ("of-syn", "gmdcsa24", "le2i", "caucafall"):
        sp.update(load_split(d))
    src_train = src[src.path.map(sp) == "train"]
    A_hop = {h: transitions_from_counts(transition_counts_timesampled(src_train, h)) for h in (0.5, 1.0)}
    prior = source_prior()

    # ---------------- lưới trên VAL
    cands = []
    for inp in inputs:
        cands.append(("baseline", inp, dict(mode="baseline", segment_hop=1.0)))
        for tau, W, hop in itertools.product((1, 2, 4, 8), (3, 5, 10), (0.5, 1.0)):
            cands.append((f"rule τ={tau},W={W},hop={hop}", inp, dict(mode="rule", tau=tau, window=W, segment_hop=hop)))
        for hop, gam, st in itertools.product((0.5, 1.0), (0.5, 1, 2, 4), (0.25, 0.5, 0.75, 1.0)):
            cands.append((f"hmm hop={hop},γ={gam},s={st}", inp,
                          dict(mode="hmm", segment_hop=hop, hmm_A=sharpen_self_loop(A_hop[hop], gam),
                               hmm_prior=prior, hmm_strength=st)))
        for s in (42, 43, 44):
            cands.append((f"hybrid v8 seed{s} hop=1.0", inp, dict(mode="hybrid", tau=4, window=10, segment_hop=1.0)))
    hybrids = {s: load_hybrid(CKPT_DIR / f"learnable_conditioner_v8_stsyn_kl0.3_seed{s}_best.pt") for s in (42, 43, 44)}
    pick = lambda name, kw: hybrids[int(name.split("seed")[1][:2])] if kw["mode"] == "hybrid" else None

    rows = []
    for name, inp, kw in tqdm(cands, desc="tuning on val"):
        m = step_metrics(val[inp], run_preds(val[inp], EngineConfig(**kw), pick(name, kw)), SEL_LAG)
        rows.append({"name": name, "input": inp, "mode": kw["mode"], **{k: round(v, 2) for k, v in m.items()},
                     "score": 0.5 * (m["fallen_f1"] + m["union_f1"])})
    grid = pd.DataFrame(rows); grid.to_csv(OUT / "stream_val_grid.csv", index=False)

    # ---------------- chọn tốt nhất mỗi họ trên val, đánh giá test
    chosen = [("baseline", "timefix")]  # mốc tham chiếu: như app cũ, không TTA không gian
    for mode in ("baseline", "rule", "hmm"):
        r = grid[grid["mode"] == mode].sort_values("score", ascending=False).iloc[0]
        chosen.append((r["name"], r["input"]))
    for s in (42, 43, 44):
        gs = grid[grid.name == f"hybrid v8 seed{s} hop=1.0"].sort_values("score", ascending=False).iloc[0]
        chosen.append((gs["name"], gs["input"]))
    chosen = list(dict.fromkeys(chosen))

    lookup = {(n, i): kw for n, i, kw in cands}
    res, ev_rows, test_preds = [], [], {}
    for name, inp in chosen:
        kw = lookup[(name, inp)]
        pv = run_preds(val[inp], EngineConfig(**kw), pick(name, kw))
        pt = run_preds(test[inp], EngineConfig(**kw), pick(name, kw))
        test_preds[f"{name} | {inp}"] = pt
        r = {"name": name, "input": inp,
             "val_score": round(grid[(grid.name == name) & (grid.input == inp)].score.iloc[0], 2)}
        for lag in LAGS:
            m = step_metrics(test[inp], pt, lag)
            r.update({f"lag{lag}_{k}": round(m[k], 2) for k in ("fallen_f1", "fall_f1", "union_f1", "bacc")})
        res.append(r)
        for on_s in (0.0, 0.5, 1.0, 1.5, 2.0):
            ev_v = event_metrics(val[inp], pv, on_s)
            ev_rows.append({"name": name, "input": inp, "alarm_on_s": on_s,
                            "val_recall_%": ev_v["episode_recall_%"], "val_fa_per_h": ev_v["false_alarms_per_hour"],
                            **event_metrics(test[inp], pt, on_s)})
    rdf, edf = pd.DataFrame(res), pd.DataFrame(ev_rows)
    rdf.to_csv(OUT / "stream_test_step.csv", index=False)
    edf.to_csv(OUT / "stream_test_events.csv", index=False)
    with open(OUT / "stream_test_preds.pkl", "wb") as f:   # cho kiểm định thống kê ở bước 8
        pickle.dump({"preds": test_preds, "streams": {i: {p: (v[0], v[2]) for p, v in test[i].items()} for i in inputs}}, f)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(rdf.to_string(index=False)); print(edf.to_string(index=False))

    # ---------------- cấu hình cho app: họ không cần huấn luyện tốt nhất trên val
    sc = lambda c: grid[(grid.name == c[0]) & (grid.input == c[1])].score.iloc[0]
    fam = [c for c in chosen if c[0].startswith(("rule", "hmm"))]
    best_name, best_inp = max(fam, key=sc)
    kw = lookup[(best_name, best_inp)]
    cfgp = OUTPUTS / "best_config.json"
    best = json.loads(cfgp.read_text(encoding="utf-8"))
    best["stream" + (f"_{TAG}" if TAG else "")] = {"name": best_name, "input": best_inp, "mode": kw["mode"], "segment_hop": kw["segment_hop"],
                      "tau": kw.get("tau", 4), "window": kw.get("window", 10),
                      "hmm_A": kw["hmm_A"].tolist() if "hmm_A" in kw else None,
                      "hmm_prior": prior.tolist(), "hmm_strength": kw.get("hmm_strength", 1.0)}
    cfgp.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Cấu hình streaming cho app:", best_name, best_inp)


if __name__ == "__main__":
    main()
