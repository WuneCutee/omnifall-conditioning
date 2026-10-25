"""Tiện ích dùng chung: load config, tính metrics, set seed."""
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


def load_config(path: str) -> dict[str, Any]:
    """Đọc file YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> dict:
    """
    Tính Sensitivity/Specificity/F1 cho 1 lớp nhị phân (vd: Fall vs. rest, Fallen vs. rest),
    đúng theo cách paper OmniFall báo cáo (Se/Sp/F1 cho từng subtask quan trọng lâm sàng).
    """
    y_true_bin = (y_true == positive_class).astype(int)
    y_pred_bin = (y_pred == positive_class).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    return {"sensitivity": sensitivity, "specificity": specificity, "f1": f1}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fall_class_id: int,
    fallen_class_id: int,
    num_classes: int = 16,
) -> dict:
    """
    Tính đầy đủ bộ metrics giống Table 2-4 trong paper OmniFall:
    Balanced Accuracy (16-class), Macro-F1, và Se/Sp/F1 cho Fall + Fallen riêng.
    """
    results = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fall": compute_binary_f1(y_true, y_pred, fall_class_id),
        "fallen": compute_binary_f1(y_true, y_pred, fallen_class_id),
    }

    # Fall ∪ Fallen (either critical state detected) — giống cột cuối trong Table 3/4 paper
    y_true_union = ((y_true == fall_class_id) | (y_true == fallen_class_id)).astype(int)
    y_pred_union = ((y_pred == fall_class_id) | (y_pred == fallen_class_id)).astype(int)
    results["fall_union_fallen"] = {
        "sensitivity": (
            ((y_true_union == 1) & (y_pred_union == 1)).sum() / max((y_true_union == 1).sum(), 1)
        ),
        "specificity": (
            ((y_true_union == 0) & (y_pred_union == 0)).sum() / max((y_true_union == 0).sum(), 1)
        ),
        "f1": f1_score(y_true_union, y_pred_union, zero_division=0),
    }

    results["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    ).tolist()

    return results


def print_metrics_table(results: dict) -> None:
    """In kết quả ra terminal theo format dễ so sánh với Table 3/4 của paper."""
    print(f"{'Metric':<20}{'Value':>10}")
    print("-" * 30)
    print(f"{'Balanced Acc':<20}{results['balanced_accuracy']*100:>9.1f}%")
    print(f"{'Macro F1':<20}{results['macro_f1']*100:>9.1f}%")
    print(f"{'Fall F1':<20}{results['fall']['f1']*100:>9.1f}%")
    print(f"{'Fallen F1':<20}{results['fallen']['f1']*100:>9.1f}%  <-- so với baseline 53.6%")
    print(f"{'Fall∪Fallen F1':<20}{results['fall_union_fallen']['f1']*100:>9.1f}%")


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
