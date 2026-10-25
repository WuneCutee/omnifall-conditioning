"""
Training loop.

Usage:
    # Bước 1: tái lập baseline
    python -m src.train --config configs/default.yaml --model baseline

    # Bước 3: model conditioning đầy đủ (sau khi pilot ở Bước 2 xác nhận hypothesis)
    python -m src.train --config configs/default.yaml --model conditioned
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline_model import BaselineClassifier, compute_class_weights
from src.conditioning import LearnableConditioner
from src.data import OmniFallDataset, VideoLevelSequenceDataset, sequence_collate_fn
from src.utils import load_config, set_seed, compute_all_metrics, print_metrics_table, ensure_dir


def train_baseline(config: dict) -> None:
    """Bước 1: tái lập baseline paper (classifier độc lập từng segment, không temporal context)."""
    set_seed(config["train"]["seed"])
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    fps = config["features"]["videomae"]["fps"]
    stride_seconds = config["features"]["videomae"]["stride_seconds"]

    train_ds = OmniFallDataset(
        sources=[tuple(s) for s in config["data"]["train_sources"]],
        feature_cache_dir=config["features"]["cache_dir"],
        fps=fps, stride_seconds=stride_seconds,
    )
    val_ds = OmniFallDataset(
        sources=[tuple(s) for s in config["data"]["val_sources"]],
        feature_cache_dir=config["features"]["cache_dir"],
        fps=fps, stride_seconds=stride_seconds,
    )

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    sample_feat_dim = train_ds[0]["feature"].shape[0]
    model = BaselineClassifier(
        feature_dim=sample_feat_dim,
        num_classes=config["data"]["taxonomy"]["num_classes"],
        hidden_dim=config["model"]["baseline"]["hidden_dim"],
        dropout=config["model"]["baseline"]["dropout"],
    ).to(device)

    all_train_labels = torch.tensor(train_ds.rows["label"])
    class_weights = compute_class_weights(
        all_train_labels, config["data"]["taxonomy"]["num_classes"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_fallen_f1 = 0.0
    patience_counter = 0
    ckpt_dir = ensure_dir("checkpoints")

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            feats = batch["feature"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["feature"].to(device)
                logits = model(feats)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        metrics = compute_all_metrics(
            y_true=torch.tensor(all_labels).numpy(),
            y_pred=torch.tensor(all_preds).numpy(),
            fall_class_id=config["data"]["taxonomy"]["fall_class_id"],
            fallen_class_id=config["data"]["taxonomy"]["fallen_class_id"],
            num_classes=config["data"]["taxonomy"]["num_classes"],
        )
        print(f"\n[Epoch {epoch+1}] loss={avg_loss:.4f}")
        print_metrics_table(metrics)

        fallen_f1 = metrics["fallen"]["f1"]
        if fallen_f1 > best_fallen_f1:
            best_fallen_f1 = fallen_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "baseline_best.pt")
            print(f"  -> Fallen-F1 cải thiện ({fallen_f1*100:.1f}%), đã lưu checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= config["train"]["early_stopping_patience"]:
                print("Early stopping.")
                break

    print(f"\n=== Kết thúc training baseline. Fallen-F1 tốt nhất: {best_fallen_f1*100:.1f}% ===")
    print(f"Train sources: {config['data']['train_sources']}, Eval sources: {config['data']['val_sources']}")
    print(
        "LƯU Ý: con số 53.6% trong paper OmniFall là cấu hình St+Sy (train) + test OF-ItW. "
        "Nếu train_sources ở trên KHÔNG bao gồm OF-Staged (vd chỉ 'of-syn'), đây KHÔNG phải "
        "phép so sánh cùng điều kiện — chỉ dùng để confirm pipeline chạy đúng."
    )


def train_conditioned(config: dict) -> None:
    """Bước 3: model conditioning đầy đủ, dùng VideoLevelSequenceDataset để có context thời gian."""
    set_seed(config["train"]["seed"])
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    fps = config["features"]["videomae"]["fps"]
    stride_seconds = config["features"]["videomae"]["stride_seconds"]

    train_ds = VideoLevelSequenceDataset(
        sources=[tuple(s) for s in config["data"]["train_sources"]],
        feature_cache_dir=config["features"]["cache_dir"],
        fps=fps, stride_seconds=stride_seconds,
    )
    train_loader = DataLoader(
        train_ds, batch_size=max(config["train"]["batch_size"] // 4, 1),
        shuffle=True, collate_fn=sequence_collate_fn,
    )

    sample_feat_dim = train_ds[0]["features"].shape[1]
    model = LearnableConditioner(
        feature_dim=sample_feat_dim,
        num_classes=config["data"]["taxonomy"]["num_classes"],
        hidden_dim=config["model"]["conditioning"]["learnable"]["hidden_dim"],
        num_heads=config["model"]["conditioning"]["learnable"]["num_attention_heads"],
        context_window_seconds=config["model"]["conditioning"]["learnable"]["context_window_seconds"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)  # -100 = padding, xem sequence_collate_fn

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [conditioned train]"):
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            starts = batch["start_times"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(feats, starts, mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f}")

    ckpt_dir = ensure_dir("checkpoints")
    torch.save(model.state_dict(), ckpt_dir / "conditioned_final.pt")
    print("Đã lưu model conditioned. Chạy src/evaluate.py để đánh giá trên OF-ItW.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["baseline", "conditioned"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model == "baseline":
        train_baseline(config)
    else:
        train_conditioned(config)
