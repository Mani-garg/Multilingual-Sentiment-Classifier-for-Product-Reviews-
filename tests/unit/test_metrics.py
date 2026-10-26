"""Unit tests for metric computations and multilingual breakdowns."""

import numpy as np

from sentiment_classifier.analysis.confusion import format_confusion_matrix
from sentiment_classifier.analysis.language_metrics import compute_multilingual_breakdown
from sentiment_classifier.analysis.metrics import calculate_all_metrics


def test_calculate_all_metrics_perfect() -> None:
    """Perfect predictions should yield 1.0 Macro-F1 and Accuracy."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    metrics = calculate_all_metrics(y_true, y_pred)
    assert np.isclose(metrics["macro_f1"], 1.0)
    assert np.isclose(metrics["accuracy"], 1.0)
    assert np.isclose(metrics["weighted_f1"], 1.0)
    assert metrics["per_class"]["positive"]["f1"] == 1.0


def test_multilingual_breakdown_flags_low_sample_count() -> None:
    """Languages with fewer than min_samples must be flagged as insufficient."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 0]
    langs = ["en", "en", "en", "en", "de", "de"]

    breakdown = compute_multilingual_breakdown(y_true, y_pred, langs, min_samples=3)
    assert breakdown["en"]["status"] == "valid"
    assert breakdown["en"]["is_statistically_sound"] is True

    # 'de' only has 2 samples (< 3)
    assert breakdown["de"]["status"] == "insufficient_samples"
    assert breakdown["de"]["is_statistically_sound"] is False


def test_format_confusion_matrix() -> None:
    """Verify confusion matrix structuring into row-wise dicts and markdown table."""
    cm = [[5, 1, 0], [2, 4, 1], [0, 1, 6]]
    res = format_confusion_matrix(cm)

    assert res["structured"]["negative"]["negative"] == 5
    assert res["structured"]["negative"]["neutral"] == 1
    assert "| **negative** | 5 | 1 | 0 | **6** |" in res["markdown_table"]
