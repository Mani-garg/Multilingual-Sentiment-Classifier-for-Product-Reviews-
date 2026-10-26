"""Per-language evaluation breakdown and statistical validity auditing."""

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from sentiment_classifier.labels import ID_TO_LABEL


def compute_multilingual_breakdown(
    y_true: list[int],
    y_pred: list[int],
    languages: list[str],
    min_samples: int = 10,
) -> dict[str, Any]:
    """Compute per-language performance statistics and flag low-sample estimates.

    Returns:
        Dictionary mapping language code to sample counts, accuracy, macro-F1,
        and validity flags.
    """
    df = pd.DataFrame(
        {
            "true": y_true,
            "pred": y_pred,
            "lang": [lang.lower() for lang in languages],
        }
    )

    total_errors = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt != yp)
    breakdown = {}

    for lang, group in df.groupby("lang"):
        g_true = group["true"].tolist()
        g_pred = group["pred"].tolist()
        count = len(g_true)
        errors = sum(1 for yt, yp in zip(g_true, g_pred, strict=False) if yt != yp)

        class_dist = {
            ID_TO_LABEL.get(k, str(k)): int(v) for k, v in group["true"].value_counts().items()
        }

        is_statistically_sound = count >= min_samples
        if is_statistically_sound:
            macro_f1 = float(f1_score(g_true, g_pred, average="macro", zero_division=0))
            accuracy = float(accuracy_score(g_true, g_pred))
        else:
            macro_f1 = None
            accuracy = None

        breakdown[lang] = {
            "sample_count": count,
            "is_statistically_sound": is_statistically_sound,
            "class_distribution": class_dist,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "error_count": errors,
            "error_percentage": float((errors / total_errors) * 100.0) if total_errors > 0 else 0.0,
            "status": "valid" if is_statistically_sound else "insufficient_samples",
        }

    return breakdown
