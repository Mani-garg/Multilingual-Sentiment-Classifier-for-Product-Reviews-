"""Report generation utilities formatting markdown summaries for human inspection."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def generate_markdown_evaluation_report(
    baseline_metrics: dict[str, Any] | None,
    transformer_metrics: dict[str, Any] | None,
    onnx_metrics: dict[str, Any] | None = None,
    error_report: dict[str, Any] | None = None,
    dataset_name: str = "Multilingual Product Reviews",
) -> str:
    """Generate a clean, structured Markdown evaluation and benchmarking report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Model Evaluation & Multilingual Performance Report",
        f"**Generated:** {timestamp}  ",
        f"**Dataset:** {dataset_name}  ",
        "**Primary Metric:** Macro-F1 (unweighted arithmetic mean across classes)  ",
        "",
        "---",
        "",
        "## 1. Global Model Comparison",
        "",
        "| Architecture / Model | Macro-F1 (%) | Accuracy (%) | Weighted-F1 (%) | Avg Latency (ms) | Status |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    models = [
        ("TF-IDF + Logistic Regression Baseline", baseline_metrics),
        ("Multilingual Transformer (DistilBERT)", transformer_metrics),
        ("ONNX Runtime Optimized Model", onnx_metrics),
    ]

    for name, m in models:
        if m:
            macro_f1 = f"{m.get('macro_f1', 0.0) * 100:.2f}%"
            acc = f"{m.get('accuracy', 0.0) * 100:.2f}%"
            weighted_f1 = f"{m.get('weighted_f1', 0.0) * 100:.2f}%"
            lat = (
                f"{m.get('avg_inference_latency_ms', m.get('avg_latency_ms_per_sample', 0.0)):.2f}"
            )
            lines.append(
                f"| **{name}** | {macro_f1} | {acc} | {weighted_f1} | {lat} ms | Verified |"
            )
        else:
            lines.append(f"| **{name}** | *N/A* | *N/A* | *N/A* | *N/A* | Pending Run |")

    lines.extend(
        [
            "",
            "## 2. Per-Class Sentiment Performance",
            "",
        ]
    )

    if transformer_metrics and "per_class" in transformer_metrics:
        lines.extend(
            [
                "| Class | Precision | Recall | F1-Score |",
                "|:---|:---:|:---:|:---:|",
            ]
        )
        for cname, cmetrics in transformer_metrics["per_class"].items():
            lines.append(
                f"| **{cname.capitalize()}** | {cmetrics['precision']:.4f} | {cmetrics['recall']:.4f} | {cmetrics['f1']:.4f} |"
            )
    elif baseline_metrics and "per_class" in baseline_metrics:
        lines.extend(
            [
                "| Class | Precision | Recall | F1-Score |",
                "|:---|:---:|:---:|:---:|",
            ]
        )
        for cname, cmetrics in baseline_metrics["per_class"].items():
            lines.append(
                f"| **{cname.capitalize()}** | {cmetrics['precision']:.4f} | {cmetrics['recall']:.4f} | {cmetrics['f1']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 3. Multilingual Evaluation Breakdown",
            "",
            "| Language | Code | Test Samples | Accuracy (%) | Macro-F1 (%) | Error Share (%) | Statistical Validity |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    active_metrics = transformer_metrics or baseline_metrics
    if active_metrics and "per_language" in active_metrics:
        for lang_code, lstats in active_metrics["per_language"].items():
            cnt = lstats.get("sample_count", 0)
            acc_val = (
                f"{lstats['accuracy'] * 100:.1f}%" if lstats.get("accuracy") is not None else "N/A"
            )
            f1_val = (
                f"{lstats['macro_f1'] * 100:.1f}%" if lstats.get("macro_f1") is not None else "N/A"
            )
            err_share = f"{lstats.get('error_percentage_of_total', lstats.get('error_percentage', 0.0)):.1f}%"
            sound = "Sufficient (>=10)" if cnt >= 10 else "Low Sample Count (<10)"
            lines.append(
                f"| {lang_code.upper()} | `{lang_code}` | {cnt} | {acc_val} | {f1_val} | {err_share} | {sound} |"
            )

    if error_report and "category_distribution" in error_report:
        lines.extend(
            [
                "",
                "## 4. Error Category Distribution",
                "",
                "| Failure Mode Category | Occurrences | Share of Misclassifications | Description |",
                "|:---|:---:|:---:|:---|",
            ]
        )
        total_errs = max(1, error_report.get("total_errors", 1))
        for cat, cnt in error_report["category_distribution"].items():
            share = (cnt / total_errs) * 100.0
            lines.append(
                f"| **{cat}** | {cnt} | {share:.1f}% | Categorized by diagnostic linguistic heuristics |"
            )

    lines.append("")
    return "\n".join(lines)


def save_markdown_report(report_md: str, path: str | Path) -> None:
    """Save markdown report to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report_md, encoding="utf-8")
