"""Confusion matrix formatting and markdown visualization."""

from typing import Any

from sentiment_classifier.labels import SUPPORTED_LABELS


def format_confusion_matrix(cm: list[list[int]], labels: list[str] | None = None) -> dict[str, Any]:
    """Format a 3x3 confusion matrix with row-wise and column-wise summaries."""
    if labels is None:
        labels = SUPPORTED_LABELS

    matrix_dict = {}
    for i, true_label in enumerate(labels):
        matrix_dict[true_label] = {}
        for j, pred_label in enumerate(labels):
            matrix_dict[true_label][pred_label] = int(cm[i][j])

    # Build ASCII / Markdown table
    header = "| True \\ Pred | " + " | ".join(labels) + " | Total |"
    divider = "|:---|:" + "---:|:".join([""] * len(labels)) + "---:|"
    rows = []
    for i, true_label in enumerate(labels):
        row_vals = [str(cm[i][j]) for j in range(len(labels))]
        total = sum(cm[i])
        rows.append(f"| **{true_label}** | " + " | ".join(row_vals) + f" | **{total}** |")

    markdown_table = "\n".join([header, divider] + rows)

    return {
        "raw_matrix": cm,
        "labels": labels,
        "structured": matrix_dict,
        "markdown_table": markdown_table,
    }
