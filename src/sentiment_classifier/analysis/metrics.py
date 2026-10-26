"""Metric calculation engine computing global and per-class performance."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sentiment_classifier.labels import ID_TO_LABEL, SUPPORTED_LABELS


def calculate_all_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Compute aggregate accuracy, macro-F1, per-class metrics, and confusion matrix."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Per-class scores
    labels = [0, 1, 2]
    class_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    class_prec = precision_score(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
    )
    class_rec = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)

    per_class = {}
    for idx in labels:
        label_name = ID_TO_LABEL.get(idx, str(idx))
        per_class[label_name] = {
            "f1": float(class_f1[idx]),
            "precision": float(class_prec[idx]),
            "recall": float(class_rec[idx]),
        }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels_order": SUPPORTED_LABELS,
    }
