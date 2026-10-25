"""
Bước 1b+2 (v8) — cải tiến v6 theo 2 hướng để giảm variance + giảm cái giá đánh đổi Fall-F1:
  1. Chọn checkpoint theo Fall_union_Fallen trên val (không chỉ Fallen-F1) — tránh chọn epoch
     "ăn gian" tăng Fallen-F1 bằng cách hy sinh Fall-F1.
  2. KL-anchor: thêm loss phụ kéo output của model về gần phân phối của rule-based (đã sửa
     lỗi, đã biết là "an toàn" — không đánh đổi Fall-F1) trong suốt quá trình train, với
     trọng số LAMBDA_KL. Ý tưởng: rule-based là 1 prior tốt, ràng buộc model không đi quá xa
     khỏi nó thay vì chỉ dùng nó làm điểm khởi đầu (residual init) như v6.

Usage:
    python -m scripts.train_learnable_conditioner_v8 --seed 42 --lambda_kl 0.3
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_model import compute_class_weights  # noqa: E402
from src.conditioning import LearnableConditioner  # noqa: E402
from src.utils import compute_all_metrics, print_metrics_table, set_seed  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "data" / "omnifall" / "rule_adjusted_logprobs"
NUM_CLASSES = 10
HIDDEN_DIM = 128
NUM_HEADS = 4
CONTEXT_WINDOW_SECONDS = 15.0
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3


class LogitSequenceDataset(Dataset):
    def __init__(self, pkl_path: Path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.items = list(data.values())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return {
            "features": torch.from_numpy(np.asarray(item["logits"], dtype=np.float32)),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "start_times": torch.tensor(item["starts"], dtype=torch.float),
        }

    def all_labels(self) -> torch.Tensor:
        return torch.cat([torch.tensor(item["labels"], dtype=torch.long) for item in self.items])


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[1]

    padded_feats = torch.zeros(len(batch), max_len, feat_dim)
    padded_labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    padded_starts = torch.zeros(len(batch), max_len, dtype=torch.float)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        padded_feats[i, :n] = item["features"]
        padded_labels[i, :n] = item["labels"]
        padded_starts[i, :n] = item["start_times"]
        mask[i, :n] = True

    return {"features": padded_feats, "labels": padded_labels, "start_times": padded_starts, "mask": mask}


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred, all_pred_raw = [], [], []
    for batch in loader:
        feats = batch["features"].to(device)
        starts = batch["start_times"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"]

        logits = model(feats, starts, mask)
        preds = logits.argmax(dim=-1).cpu()
        preds_raw = feats.argmax(dim=-1).cpu()

        valid = mask.cpu()
        all_true.extend(labels[valid].numpy())
        all_pred.extend(preds[valid].numpy())
        all_pred_raw.extend(preds_raw[valid].numpy())
    return np.array(all_true), np.array(all_pred), np.array(all_pred_raw)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_kl", type=float, default=0.3,
                         help="Trọng số KL-anchor kéo output về gần rule-based (0 = tắt, giống v6)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, seed: {args.seed}")

    train_ds = LogitSequenceDataset(CACHE_DIR / "stsyn_train.pkl")
    oops_val_ds = LogitSequenceDataset(CACHE_DIR / "OOPS_val_stsyn.pkl")
    test_ds = LogitSequenceDataset(CACHE_DIR / "OOPS_test_stsyn.pkl")
    print(f"train videos: {len(train_ds)}, OOPS val videos: {len(oops_val_ds)}, test videos: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    oops_val_loader = DataLoader(oops_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    class_weights = compute_class_weights(train_ds.all_labels(), num_classes=NUM_CLASSES).to(device)
    print("Class weights:", class_weights.cpu().numpy().round(2))

    model = LearnableConditioner(
        feature_dim=NUM_CLASSES, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS, context_window_seconds=CONTEXT_WINDOW_SECONDS, residual=True,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # v8: chọn checkpoint theo Fall∪Fallen (không phải chỉ Fallen-F1) — tránh việc chọn epoch
    # "ăn gian" tăng Fallen-F1 bằng cách hy sinh Fall-F1 (đã thấy rõ ở v6: variance cao +
    # luôn đánh đổi Fall-F1). Fall∪Fallen là metric quan trọng nhất về mặt lâm sàng, tối ưu
    # trực tiếp nó khi chọn checkpoint nên cho kết quả cân bằng hơn.
    best_val_score = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            starts = batch["start_times"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(feats, starts, mask)
            ce_loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

            # KL-anchor: kéo phân phối output về gần rule-based (feats đã là log-probs của
            # rule-based, xem cache_rule_adjusted_logprobs*.py). Chỉ tính trên vị trí hợp lệ.
            if args.lambda_kl > 0:
                rule_probs = torch.exp(feats)  # feats = log(rule_probs + eps)
                rule_probs = rule_probs / rule_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                log_student = F.log_softmax(logits, dim=-1)
                kl = F.kl_div(log_student, rule_probs, reduction="none").sum(dim=-1)  # (B, L)
                kl = (kl * mask.float()).sum() / mask.float().sum().clamp(min=1)
                loss = ce_loss + args.lambda_kl * kl
            else:
                loss = ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        y_true_val, y_pred_val, _ = evaluate(model, oops_val_loader, device)
        m_val = compute_all_metrics(y_true_val, y_pred_val, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
        val_fallen_f1 = m_val["fallen"]["f1"]
        val_score = m_val["fall_union_fallen"]["f1"]
        print(f"[Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f} "
              f"OOPS_val_fallen_f1={val_fallen_f1*100:.2f}% OOPS_val_fall_union_fallen={val_score*100:.2f}%")

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest OOPS-val Fall_union_Fallen: {best_val_score*100:.2f}% (epoch {best_epoch}) -- loading for test eval.")
    model.load_state_dict(best_state)

    y_true_test, y_pred_test, y_pred_test_raw = evaluate(model, test_loader, device)

    print("\n=== RULE-BASED (nền, chưa qua learnable v4) trên OOPS test ===")
    m_raw = compute_all_metrics(y_true_test, y_pred_test_raw, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
    print_metrics_table(m_raw)

    print("\n=== HYBRID v8 (chọn theo Fall_union_Fallen + KL-anchor) (St+Sy baseline, residual trên rule-based, class-balanced + dropout) — OOPS test ===")
    m_test = compute_all_metrics(y_true_test, y_pred_test, fall_class_id=1, fallen_class_id=2, num_classes=NUM_CLASSES)
    print_metrics_table(m_test)

    delta = (m_test["fallen"]["f1"] - m_raw["fallen"]["f1"]) * 100
    print(f"\n>>> Thay đổi Fallen-F1 so với RULE-BASED (cùng cache, cùng segment): {delta:+.2f} điểm "
          f"({m_raw['fallen']['f1']*100:.2f}% -> {m_test['fallen']['f1']*100:.2f}%)")

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = f"learnable_conditioner_v8_stsyn_kl{args.lambda_kl}_seed{args.seed}_best.pt"
    torch.save(best_state, ckpt_dir / ckpt_name)
    print(f"\nĐã lưu checkpoint: {ckpt_dir / ckpt_name}")


if __name__ == "__main__":
    main()
