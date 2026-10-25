"""
Đánh giá model trên OF-ItW, đúng protocol paper (Se/Sp/F1 cho Fall & Fallen riêng).

Usage:
    python -m src.evaluate --config configs/default.yaml \\
        --checkpoint checkpoints/baseline_best.pt --model-type baseline

    python -m src.evaluate --config configs/default.yaml \\
        --checkpoint checkpoints/conditioned_final.pt --model-type conditioned
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baseline_model import BaselineClassifier
from src.conditioning import LearnableConditioner, RuleBasedConditioner
from src.data import OmniFallDataset, VideoLevelSequenceDataset, sequence_collate_fn
from src.utils import load_config, compute_all_metrics, print_metrics_table


def evaluate_baseline(config: dict, checkpoint_path: str) -> dict:
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    test_ds = OmniFallDataset(
        sources=[tuple(s) for s in config["data"]["eval_sources"]],
        feature_cache_dir=config["features"]["cache_dir"],
        fps=config["features"]["videomae"]["fps"],
        stride_seconds=config["features"]["videomae"]["stride_seconds"],
    )
    loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    sample_feat_dim = test_ds[0]["feature"].shape[0]
    model = BaselineClassifier(
        feature_dim=sample_feat_dim,
        num_classes=config["data"]["taxonomy"]["num_classes"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["feature"].to(device)
            logits = model(feats)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    return compute_all_metrics(
        y_true=np.array(all_labels), y_pred=np.array(all_preds),
        fall_class_id=config["data"]["taxonomy"]["fall_class_id"],
        fallen_class_id=config["data"]["taxonomy"]["fallen_class_id"],
        num_classes=config["data"]["taxonomy"]["num_classes"],
    )


def evaluate_conditioned(config: dict, checkpoint_path: str) -> dict:
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    test_ds = VideoLevelSequenceDataset(
        sources=[tuple(s) for s in config["data"]["eval_sources"]],
        feature_cache_dir=config["features"]["cache_dir"],
        fps=config["features"]["videomae"]["fps"],
        stride_seconds=config["features"]["videomae"]["stride_seconds"],
    )
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=sequence_collate_fn)

    sample_feat_dim = test_ds[0]["features"].shape[1]
    model = LearnableConditioner(
        feature_dim=sample_feat_dim,
        num_classes=config["data"]["taxonomy"]["num_classes"],
        hidden_dim=config["model"]["conditioning"]["learnable"]["hidden_dim"],
        num_heads=config["model"]["conditioning"]["learnable"]["num_attention_heads"],
        context_window_seconds=config["model"]["conditioning"]["learnable"]["context_window_seconds"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            starts = batch["start_times"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]

            logits = model(feats, starts, mask)
            preds = logits.argmax(dim=-1).cpu()

            valid = mask.cpu()
            all_preds.extend(preds[valid].numpy())
            all_labels.extend(labels[valid].numpy())

    return compute_all_metrics(
        y_true=np.array(all_labels), y_pred=np.array(all_preds),
        fall_class_id=config["data"]["taxonomy"]["fall_class_id"],
        fallen_class_id=config["data"]["taxonomy"]["fallen_class_id"],
        num_classes=config["data"]["taxonomy"]["num_classes"],
    )


def evaluate_rule_based_pilot(config: dict, baseline_checkpoint: str) -> dict:
    """
    Bước 2: áp dụng RuleBasedConditioner LÊN TRÊN baseline đã train sẵn (post-hoc),
    không cần train lại gì — đây là cách nhanh nhất để kiểm tra hypothesis.
    """
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    test_ds = VideoLevelSequenceDataset(
        sources=[tuple(s) for s in config["data"]["eval_sources"]],
        feature_cache_dir=config["features"]["cache_dir"],
        fps=config["features"]["videomae"]["fps"],
        stride_seconds=config["features"]["videomae"]["stride_seconds"],
    )
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=sequence_collate_fn)

    sample_feat_dim = test_ds[0]["features"].shape[1]
    baseline = BaselineClassifier(
        feature_dim=sample_feat_dim, num_classes=config["data"]["taxonomy"]["num_classes"],
    ).to(device)
    baseline.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
    baseline.eval()

    rule_cfg = config["model"]["conditioning"]["rule_based"]
    conditioner = RuleBasedConditioner(
        boost_window_seconds=rule_cfg["boost_window_seconds"],
        decay_halflife_seconds=rule_cfg["decay_halflife_seconds"],
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            starts = batch["start_times"]
            mask = batch["mask"]
            labels = batch["labels"]

            batch_size, seq_len, feat_dim = feats.shape
            logits = baseline(feats.reshape(-1, feat_dim)).reshape(batch_size, seq_len, -1)
            probs = torch.softmax(logits, dim=-1).cpu()

            adjusted_probs = conditioner.apply(probs, starts, mask)
            preds = adjusted_probs.argmax(dim=-1)

            valid = mask
            all_preds.extend(preds[valid].numpy())
            all_labels.extend(labels[valid].numpy())

    return compute_all_metrics(
        y_true=np.array(all_labels), y_pred=np.array(all_preds),
        fall_class_id=config["data"]["taxonomy"]["fall_class_id"],
        fallen_class_id=config["data"]["taxonomy"]["fallen_class_id"],
        num_classes=config["data"]["taxonomy"]["num_classes"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str,
                         choices=["baseline", "conditioned", "rule_based_pilot"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model_type == "baseline":
        metrics = evaluate_baseline(config, args.checkpoint)
    elif args.model_type == "conditioned":
        metrics = evaluate_conditioned(config, args.checkpoint)
    else:
        metrics = evaluate_rule_based_pilot(config, args.checkpoint)

    print(f"\n=== Kết quả đánh giá trên {config['data']['eval_sources']} ===")
    print_metrics_table(metrics)
