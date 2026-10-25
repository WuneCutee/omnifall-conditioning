"""Bước 5 — Huấn luyện lại Hybrid conditioner (residual attention + KL-anchor) có kiểm soát:

  * lưới λ_KL ∈ {0, 0.1, 0.3, 0.5, 1.0} x 3 seed  -> có ablation thật cho KL-anchor
    (λ=0 + chọn checkpoint theo Fall∪Fallen = "chỉ composite selection")
  * 2 chế độ dữ liệu: "stsyn" (đúng protocol staged->wild) và "stsyn+itw"
    (thêm 165 video ItW-train có nhãn — cấu hình thích nghi miền, báo cáo RIÊNG)
  * đầu vào = log xác suất sau rule-conditioning (τ, W lấy từ best_config.json)
  * chọn checkpoint theo Fall∪Fallen trên ItW-val; chọn λ theo điểm val trung bình 3 seed.

Usage:
    python pipeline/s05_train_hybrid.py [--epochs 40]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax

from tcfall.config import DATA, NUM_CLASSES, OUTPUTS
from tcfall.conditioners import hmm_filter, prior_shift, rule_condition
from tcfall.metrics import core_metrics
from tcfall.study import load_logits, load_with_spatial, video_probs
from src.conditioning import LearnableConditioner
from src.baseline_model import compute_class_weights

OUT = OUTPUTS / "hybrid"
EPS = 1e-6


def make_items(prob_seqs):
    """[(P (T,C), labels, starts)] -> list dict tensor với đầu vào = log(rule(P))."""
    return [{"x": torch.tensor(np.log(P + EPS), dtype=torch.float32),
             "y": torch.tensor(np.asarray(y), dtype=torch.long),
             "t": torch.tensor(np.asarray(s), dtype=torch.float32)} for P, y, s in prob_seqs]


def collate(batch):
    L = max(b["x"].shape[0] for b in batch)
    x = torch.zeros(len(batch), L, NUM_CLASSES); y = torch.full((len(batch), L), -100, dtype=torch.long)
    t = torch.zeros(len(batch), L); m = torch.zeros(len(batch), L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["x"].shape[0]
        x[i, :n], y[i, :n], t[i, :n], m[i, :n] = b["x"], b["y"], b["t"], True
    return x, y, t, m


@torch.no_grad()
def predict(model, items, device):
    model.eval()
    ys, ps = [], []
    for i in range(0, len(items), 64):
        x, y, t, m = [z.to(device) for z in collate(items[i:i + 64])]
        p = model(x, t, m).argmax(-1)
        ys.append(y[m].cpu().numpy()); ps.append(p[m].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def per_video_preds(model, items, device):
    model.eval(); out = []
    with torch.no_grad():
        for it in items:
            x, y, t, m = [z.to(device) for z in collate([it])]
            out.append((it["y"].numpy(), model(x, t, m).argmax(-1)[0].cpu().numpy()))
    return out


def train_one(train_items, val_items, lam, seed, epochs, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LearnableConditioner(NUM_CLASSES, NUM_CLASSES, 128, 4, 15.0, residual=True, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    w = compute_class_weights(torch.cat([it["y"] for it in train_items]), num_classes=NUM_CLASSES).to(device)
    ce = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    best, best_state, best_ep = -1, None, -1
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(train_items), generator=g).tolist()
        for i in range(0, len(order), 32):
            x, y, t, m = [z.to(device) for z in collate([train_items[j] for j in order[i:i + 32]])]
            logits = model(x, t, m)
            loss = ce(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            if lam > 0:
                rp = torch.exp(x); rp = rp / rp.sum(-1, keepdim=True).clamp(min=1e-8)
                kl = F.kl_div(F.log_softmax(logits, -1), rp, reduction="none").sum(-1)
                loss = loss + lam * (kl * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        mv = core_metrics(*predict(model, val_items, device))
        if mv["union_f1"] > best:
            best, best_ep = mv["union_f1"], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lambdas", default="0,0.1,0.3,0.5,1.0")
    ap.add_argument("--input", choices=["rule", "hmm"], default="rule",
                    help="prior mà Hybrid học phần dư lên trên: rule-conditioning hoặc HMM lọc tiến")
    ap.add_argument("--data_modes", default="stsyn,stsyn+itw")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--paper_setup", action="store_true",
                    help="đúng thiết lập bài báo: feature legacy, crop giữa, rule τ=4 W=10 (ablation λ_KL)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    best = json.loads((OUTPUTS / "best_config.json").read_text(encoding="utf-8"))
    tau, W, variant, crops = best["tau"], best["window"], best["variant"], best["crops"]
    A, hp, hs = np.array(best["hmm_A"]), np.array(best["hmm_prior"]), best["hmm_strength"]
    if args.paper_setup:
        tau, W, variant, crops = 4.0, 10.0, "legacy", "center"
        best = {**best, "prior_alpha": 0.0}
    args.input_tag = args.input + (f"_{args.tag}" if args.tag else "")

    def prior_cond(P, starts):
        if args.input == "hmm":
            return hmm_filter(P, A, hp, hs)
        return rule_condition(P, np.asarray(starts), tau, W)

    # Dữ liệu train nguồn (St+Sy): logits baseline cũ (feature đã bị xoá nên không tính lại được)
    with open(DATA / "baseline_logits" / "stsyn_train.pkl", "rb") as f:
        src = pickle.load(f)
    src_items = make_items([(prior_cond(softmax(np.asarray(v["logits"]), -1), v["starts"]),
                             v["labels"], v["starts"]) for v in src.values()])

    def itw_items(split):
        items = load_with_spatial(split) if variant == "spatial" else load_logits(variant, split)
        seqs = []
        for k in sorted(items):
            it = items[k]
            P = video_probs(it, crops)
            if best.get("prior_alpha", 0) > 0 and "ItW-train" in best["prior"]:
                P = prior_shift(P, np.array(best["src_prior"]), np.array(best["itw_prior"]), best["prior_alpha"])
            seqs.append((prior_cond(P, it["starts"]), it["labels"], it["starts"]))
        return make_items(seqs)

    itw_tr, val, test = itw_items("train"), itw_items("val"), itw_items("test")
    rows = []
    for data_mode in args.data_modes.split(","):
        train = src_items + (itw_tr * 5 if data_mode == "stsyn+itw" else [])  # x5: cân bằng với 9877 video nguồn
        for lam in [float(x) for x in args.lambdas.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                t0 = time.time()
                model, ep = train_one(train, val, lam, seed, args.epochs, device)
                mv = core_metrics(*predict(model, val, device))
                mt = core_metrics(*predict(model, test, device))
                name = f"hyb_{args.input_tag}_{data_mode.replace('+', '_')}_kl{lam}_seed{seed}"
                torch.save(model.state_dict(), OUT / f"{name}.pt")
                with open(OUT / f"{name}_test_preds.pkl", "wb") as f:
                    pickle.dump(per_video_preds(model, test, device), f)
                rows.append({"input": args.input_tag, "data": data_mode, "lambda": lam, "seed": seed, "best_epoch": ep,
                             **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}})
                print(f"{name}: ep={ep} val fallen={mv['fallen_f1']:.1f} union={mv['union_f1']:.1f} | "
                      f"test fallen={mt['fallen_f1']:.1f} fall={mt['fall_f1']:.1f} union={mt['union_f1']:.1f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(OUT / f"hybrid_grid_{args.input_tag}.csv", index=False)

    df = pd.DataFrame(rows)
    df["val_score"] = 0.5 * (df.val_fallen_f1 + df.val_union_f1)
    agg = df.groupby(["input", "data", "lambda"]).agg(
        val_score=("val_score", "mean"), test_fallen=("test_fallen_f1", "mean"), test_fallen_sd=("test_fallen_f1", "std"),
        test_fall=("test_fall_f1", "mean"), test_fall_sd=("test_fall_f1", "std"),
        test_union=("test_union_f1", "mean"), test_union_sd=("test_union_f1", "std")).round(2).reset_index()
    agg.to_csv(OUT / f"hybrid_grid_summary_{args.input_tag}.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
