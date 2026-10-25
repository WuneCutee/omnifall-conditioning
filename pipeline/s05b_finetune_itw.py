"""Bước 5b — Thích nghi miền có nhãn: fine-tune classifier baseline (FeatureTransformer) trên
165 video ItW-TRAIN (809 segment). Tương ứng cấu hình St+Sy+ItW của OmniFall — BÁO CÁO RIÊNG,
không trộn với protocol staged->wild thuần.

Tiền xử lý / tăng cường dữ liệu ở mức feature (không cần giải mã lại video):
  * crop thời gian ngẫu nhiên: vị trí 18 feature trong segment u ~ U(0,1) (giống code gốc)
  * crop không gian ngẫu nhiên: feature của crop trái / giữa / phải (đã trích ở bước 2, 2b)
  * label smoothing 0.1 (như lúc huấn luyện baseline gốc)
Chống quên: L2-SP — phạt ||θ - θ_gốc||² (Li et al., 2018), vì feature nguồn đã bị xoá nên
không trộn được dữ liệu St+Sy.

Lưới (lr, λ_L2SP) chọn trên ItW-val (điểm = (Fallen-F1 + Fall∪Fallen-F1)/2, crop giữa);
cấu hình tốt nhất chạy 3 seed; lưu checkpoint dùng được cho app (--baseline_ckpt).

Usage:
    python pipeline/s05b_finetune_itw.py
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tcfall.baseline import load_baseline
from tcfall.config import NUM_CLASSES, OUTPUTS, feature_dir
from tcfall.data import clean_labels, load_labels, load_split
from tcfall.metrics import core_metrics
from tcfall.segments import crop_positions, gather, window_start

OUT = OUTPUTS / "finetune"
CROP_VARIANTS = ("timefix", "timefix_start", "timefix_end")


def load_split_data(split):
    labels, _ = clean_labels(load_labels("OOPS"))
    sp = load_split("OOPS")
    sub = labels[labels.path.map(sp) == split]
    vids = []
    for path, g in sub.groupby("path", sort=True):
        g = g.sort_values("start")
        feats = {v: np.load(feature_dir("OOPS", v) / f"{Path(path).name}.npy").astype(np.float32) for v in CROP_VARIANTS}
        vids.append({"path": path, "feats": feats, "start": g.start.to_numpy(), "end": g.end.to_numpy(),
                     "label": g.label.to_numpy()})
    return vids


def sample_batch(vids, rng, augment: bool):
    X, Y = [], []
    for v in vids:
        for s, e, y in zip(v["start"], v["end"], v["label"]):
            var = CROP_VARIANTS[rng.integers(3)] if augment else "timefix"
            f = v["feats"][var]
            u = rng.random() if augment else 0.5
            X.append(gather(f, window_start(s, e, u, len(f)))); Y.append(y)
    return torch.from_numpy(np.stack(X)), torch.tensor(Y)


@torch.no_grad()
def predict(model, vids, device, spatial=False, k_temporal=1):
    """Dự đoán theo segment; spatial=True: trung bình 3 crop không gian; k_temporal: TTA thời gian."""
    model.eval()
    ys, ps = [], []
    per_video = []
    for v in vids:
        P = 0
        variants = CROP_VARIANTS if spatial else ("timefix",)
        for var in variants:
            f = v["feats"][var]
            for u in crop_positions(k_temporal):
                X = torch.from_numpy(np.stack([gather(f, window_start(s, e, u, len(f)))
                                               for s, e in zip(v["start"], v["end"])])).to(device)
                P = P + torch.softmax(model(X)["logits"].float(), -1).cpu().numpy()
        pred = P.argmax(1)
        per_video.append((v["label"], pred))
        ys.append(v["label"]); ps.append(pred)
    return core_metrics(np.concatenate(ys), np.concatenate(ps)), per_video


def train(base_state, train_vids, val_vids, lr, l2sp, seed, device, epochs=30, bs=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = load_baseline(device)
    model.load_state_dict(base_state)
    ref = {k: v.detach().clone() for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        X, Y = sample_batch(train_vids, rng, augment=True)
        perm = torch.randperm(len(Y), generator=torch.Generator().manual_seed(seed * 1000 + ep))
        for i in range(0, len(Y), bs):
            idx = perm[i:i + bs]
            logits = model(X[idx].to(device))["logits"]
            loss = F.cross_entropy(logits, Y[idx].to(device), label_smoothing=0.1)
            if l2sp > 0:
                loss = loss + l2sp * sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        m, _ = predict(model, val_vids, device)
        sc = 0.5 * (m["fallen_f1"] + m["union_f1"])
        if sc > best:
            best, best_ep, best_state = sc, ep + 1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best, best_ep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = load_split_data("train"), load_split_data("val"), load_split_data("test")
    base = load_baseline(device)
    base_state = copy.deepcopy(base.state_dict())
    m0, _ = predict(base, va, device)
    print(f"Baseline gốc trên val: fallen={m0['fallen_f1']:.1f} union={m0['union_f1']:.1f}")

    rows = []
    for lr, l2sp in itertools.product((1e-5, 3e-5, 1e-4, 2e-4, 5e-4), (0.0, 1e-3, 1e-2)):
        _, sc, ep = train(base_state, tr, va, lr, l2sp, 42, device)
        rows.append({"lr": lr, "l2sp": l2sp, "val_score": round(sc, 2), "best_epoch": ep})
        print(rows[-1], flush=True)
    grid = pd.DataFrame(rows); grid.to_csv(OUT / "finetune_grid_val.csv", index=False)
    b = grid.loc[grid.val_score.idxmax()]
    lr, l2sp = float(b.lr), float(b.l2sp)
    print(f"Chọn lr={lr}, l2sp={l2sp}")

    res, preds = [], {}
    for seed in (42, 43, 44):
        model, sc, ep = train(base_state, tr, va, lr, l2sp, seed, device)
        ck = OUT / f"baseline_ft_itw_seed{seed}.pt"
        torch.save(model.state_dict(), ck)
        for spatial, k in ((False, 1), (True, 1), (False, 5), (True, 5)):
            mv, _ = predict(model, va, device, spatial, k)
            mt, pv = predict(model, te, device, spatial, k)
            tag = f"{'spatial' if spatial else 'center'}_tta{k}"
            preds[f"seed{seed}_{tag}"] = pv
            res.append({"seed": seed, "eval": tag, "best_epoch": ep, **{f"val_{x}": round(y, 2) for x, y in mv.items()},
                        **{f"test_{x}": round(y, 2) for x, y in mt.items()}})
            print(res[-1], flush=True)
    df = pd.DataFrame(res); df.to_csv(OUT / "finetune_results.csv", index=False)
    summ = df.groupby("eval")[[c for c in df if c.startswith(("val_", "test_"))]].agg(["mean", "std"]).round(2)
    summ.to_csv(OUT / "finetune_summary.csv")
    print(summ[[("val_fallen_f1", "mean"), ("test_fallen_f1", "mean"), ("test_fallen_f1", "std"),
                ("test_fall_f1", "mean"), ("test_union_f1", "mean"), ("test_union_f1", "std")]])
    import pickle
    with open(OUT / "finetune_test_preds.pkl", "wb") as f:
        pickle.dump(preds, f)
    (OUT / "finetune_config.json").write_text(json.dumps({"lr": lr, "l2sp": l2sp}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
