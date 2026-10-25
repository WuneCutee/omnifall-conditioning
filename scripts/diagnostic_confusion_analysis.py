"""
Phân tích chẩn đoán sâu — trả lời 3 câu hỏi khoa học:
  1. Fall-F1 giảm ở đâu? (baseline -> rule-based) — confusion matrix chi tiết cho true=fall.
  2. Fallen-F1 tăng nhờ sửa nhầm lẫn với LỚP NÀO? (fallen<->lying hay fallen<->class khác)
  3. Rule-based có thật sự "cướp" xác suất từ fall sang fallen do segment liên tiếp không?
     (kiểm tra trực tiếp: trong các segment true=fall bị rule-based đổi dự đoán, khoảng cách
     thời gian tới lần "fall" được dự đoán gần nhất trước đó trong CÙNG video là bao nhiêu?)

Usage:
    python -m scripts.diagnostic_confusion_analysis
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conditioning import LearnableConditioner, RuleBasedConditioner, TRANSIENT_TO_STATIC  # noqa: E402

BASELINE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "baseline_logits" / "OOPS_test_stsyn.pkl"
RULE_CACHE = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs" / "OOPS_test_stsyn.pkl"
V5_CKPT = PROJECT_ROOT / "checkpoints" / "learnable_conditioner_v5_stsyn_seed42_best.pt"

CLASS_NAMES = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
               "stand_up", "standing", "other"]
NUM_CLASSES = 10


def load_cache_preds(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    y_true, y_pred = [], []
    video_ids, starts_all = [], []
    for key, item in data.items():
        preds = np.argmax(item["logits"], axis=-1)
        y_true.extend(item["labels"])
        y_pred.extend(preds)
        video_ids.extend([key] * len(item["labels"]))
        starts_all.extend(item["starts"])
    return np.array(y_true), np.array(y_pred), video_ids, np.array(starts_all)


@torch.no_grad()
def load_v5_preds():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LearnableConditioner(
        feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=128, num_heads=4,
        context_window_seconds=15.0, residual=True, dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(V5_CKPT, map_location=device, weights_only=True))
    model.eval()

    with open(RULE_CACHE, "rb") as f:
        data = pickle.load(f)

    y_true, y_pred, video_ids, starts_all = [], [], [], []
    for key, item in data.items():
        feats = torch.tensor(np.asarray(item["logits"], dtype=np.float32)).unsqueeze(0).to(device)
        starts = torch.tensor([item["starts"]], dtype=torch.float).to(device)
        mask = torch.ones(1, feats.shape[1], dtype=torch.bool).to(device)
        logits = model(feats, starts, mask)
        preds = logits.argmax(dim=-1)[0].cpu().numpy()
        y_true.extend(item["labels"])
        y_pred.extend(preds)
        video_ids.extend([key] * len(item["labels"]))
        starts_all.extend(item["starts"])
    return np.array(y_true), np.array(y_pred), video_ids, np.array(starts_all)


def print_confusion_for_true_class(y_true, y_pred, true_class: int, name: str, label: str):
    mask = y_true == true_class
    n = mask.sum()
    if n == 0:
        print(f"  [{label}] true={name}: KHÔNG CÓ mẫu")
        return
    preds = y_pred[mask]
    counts = np.bincount(preds, minlength=NUM_CLASSES)
    parts = [f"{CLASS_NAMES[i]}={counts[i]}({100*counts[i]/n:.1f}%)" for i in range(NUM_CLASSES) if counts[i] > 0]
    print(f"  [{label}] true={name} (n={n}): " + ", ".join(parts))


def main():
    print("=" * 150)
    print("PHẦN 1 — Confusion breakdown cho true=fall và true=fallen (baseline vs rule-based vs hybrid v5)")
    print("=" * 150)

    y_true_b, y_pred_b, vid_b, starts_b = load_cache_preds(BASELINE_CACHE)
    y_true_r, y_pred_r, vid_r, starts_r = load_cache_preds(RULE_CACHE)
    y_true_h, y_pred_h, vid_h, starts_h = load_v5_preds()

    for true_class, name in [(1, "fall"), (2, "fallen"), (6, "lying")]:
        print(f"\n--- true = {name} ---")
        print_confusion_for_true_class(y_true_b, y_pred_b, true_class, name, "Baseline")
        print_confusion_for_true_class(y_true_r, y_pred_r, true_class, name, "Rule-based")
        print_confusion_for_true_class(y_true_h, y_pred_h, true_class, name, "Hybrid v5")

    print("\n" + "=" * 150)
    print("PHẦN 2 — Trong các segment true=fall bị Rule-based đổi dự đoán so với Baseline: đổi THÀNH gì?")
    print("=" * 150)
    # Ghép theo (video_id, start) để so sánh đúng cặp baseline vs rule-based (cùng segment)
    key_to_idx_b = {(v, round(s, 3)): i for i, (v, s) in enumerate(zip(vid_b, starts_b))}
    n_fall_total = 0
    n_changed = 0
    changed_to = []
    for i in range(len(y_true_r)):
        if y_true_r[i] != 1:
            continue
        n_fall_total += 1
        key = (vid_r[i], round(starts_r[i], 3))
        j = key_to_idx_b.get(key)
        if j is None:
            continue
        if y_pred_b[j] == 1 and y_pred_r[i] != 1:
            n_changed += 1
            changed_to.append(y_pred_r[i])
    print(f"Tổng segment true=fall: {n_fall_total}")
    print(f"Baseline dự đoán ĐÚNG (fall) nhưng Rule-based đổi SAI: {n_changed} "
          f"({100*n_changed/max(n_fall_total,1):.1f}% tổng số true=fall)")
    if changed_to:
        counts = np.bincount(changed_to, minlength=NUM_CLASSES)
        for i in range(NUM_CLASSES):
            if counts[i] > 0:
                print(f"    -> đổi thành {CLASS_NAMES[i]}: {counts[i]} ({100*counts[i]/n_changed:.1f}%)")

    print("\n" + "=" * 150)
    print("PHẦN 3 — Trong các segment bị đổi fall->fallen: khoảng cách tới lần 'fall' được dự đoán TRƯỚC ĐÓ")
    print("(giả thuyết: đây là fall-segment THỨ 2 liên tiếp, bị rule 'ăn theo' fall-segment đầu tiên)")
    print("=" * 150)
    # Với mỗi video, sort theo start, tìm các segment baseline dự đoán fall liên tiếp
    from collections import defaultdict
    video_segments = defaultdict(list)
    for i in range(len(y_true_r)):
        video_segments[vid_r[i]].append((starts_r[i], y_true_r[i], y_pred_b[key_to_idx_b.get((vid_r[i], round(starts_r[i],3)), i)] if (vid_r[i], round(starts_r[i],3)) in key_to_idx_b else -1, y_pred_r[i]))

    gaps = []
    for vid, segs in video_segments.items():
        segs.sort(key=lambda x: x[0])
        last_fall_pred_time = None
        for start, true_l, base_pred, rule_pred in segs:
            if true_l == 1 and base_pred == 1 and rule_pred != 1:
                if last_fall_pred_time is not None:
                    gaps.append(start - last_fall_pred_time)
            if base_pred == 1:
                last_fall_pred_time = start
    if gaps:
        gaps = np.array(gaps)
        print(f"Số segment 'fall bị đổi sai' có 1 fall-prediction TRƯỚC ĐÓ trong cùng video: {len(gaps)}")
        print(f"Khoảng cách thời gian (giây) tới fall-prediction trước đó: "
              f"mean={gaps.mean():.2f}s, median={np.median(gaps):.2f}s, "
              f"trong vòng 10s (boost window): {100*(gaps<=10).mean():.1f}%")
    else:
        print("Không tìm thấy segment nào khớp mẫu này.")

    print("\n" + "=" * 150)
    print("PHẦN 4 — Paired test: Rule-based vs Hybrid v5 trên cùng segment (McNemar cho 'dự đoán đúng fallen')")
    print("=" * 150)
    # Ghép rule-based và hybrid v5 theo (video, start) — cả 2 đều chạy trên RULE_CACHE nên cùng thứ tự
    correct_r = (y_true_r == 2) & (y_pred_r == 2)
    wrong_r_should_be_fallen = (y_true_r == 2) & (y_pred_r != 2)
    correct_h = (y_true_h == 2) & (y_pred_h == 2)

    # Chỉ xét các segment true=fallen (đây là nhóm quan trọng nhất)
    mask_fallen = y_true_r == 2
    r_correct = (y_pred_r == 2)[mask_fallen]
    h_correct = (y_pred_h == 2)[mask_fallen]

    both_correct = int((r_correct & h_correct).sum())
    only_r = int((r_correct & ~h_correct).sum())
    only_h = int((~r_correct & h_correct).sum())
    both_wrong = int((~r_correct & ~h_correct).sum())
    print(f"Trong {mask_fallen.sum()} segment true=fallen:")
    print(f"  Cả 2 đúng: {both_correct}, chỉ Rule đúng: {only_r}, chỉ Hybrid đúng: {only_h}, cả 2 sai: {both_wrong}")

    # McNemar's test (không cần continuity correction cho mẫu vừa phải)
    if only_r + only_h > 0:
        from scipy.stats import chi2
        stat = (abs(only_r - only_h) - 1) ** 2 / (only_r + only_h) if (only_r + only_h) > 0 else 0
        p_value = 1 - chi2.cdf(stat, df=1)
        print(f"  McNemar chi2 (continuity-corrected) = {stat:.3f}, p-value = {p_value:.4f}")
        print(f"  {'=> CÓ ý nghĩa thống kê (p<0.05)' if p_value < 0.05 else '=> KHÔNG có ý nghĩa thống kê (p>=0.05) -- Hybrid v5 chưa chứng minh vượt Rule-based rõ ràng'}")
    else:
        print("  Không đủ cặp lệch nhau để chạy McNemar.")


if __name__ == "__main__":
    main()
