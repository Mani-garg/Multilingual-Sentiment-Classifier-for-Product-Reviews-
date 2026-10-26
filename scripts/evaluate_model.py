"""Evaluate model performance on specified dataset partitions."""

import argparse
from pathlib import Path

from sentiment_classifier.analysis.metrics import calculate_all_metrics
from sentiment_classifier.analysis.reports import (
    generate_markdown_evaluation_report,
    save_markdown_report,
)
from sentiment_classifier.data.loading import load_dataset
from sentiment_classifier.inference.predictor import SentimentPredictor
from sentiment_classifier.labels import LABEL_TO_ID
from sentiment_classifier.utils.serialization import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier on partition.")
    parser.add_argument("--split-path", type=str, default="data/processed/test.jsonl")
    parser.add_argument(
        "--backend", type=str, default="baseline", choices=["baseline", "onnx", "quantized_onnx"]
    )
    parser.add_argument(
        "--metrics-out", type=str, default="artifacts/reports/evaluation_metrics.json"
    )
    parser.add_argument("--report-out", type=str, default="artifacts/reports/evaluation_report.md")
    args = parser.parse_args()

    print(f"Loading evaluation dataset: {args.split_path}")
    df = load_dataset(args.split_path)

    texts = df["review_text"].tolist()
    true_labels = df["sentiment_label"].tolist()
    y_true = [LABEL_TO_ID[lbl] for lbl in true_labels]
    languages = df["language"].tolist() if "language" in df.columns else None

    # Load predictor
    from sentiment_classifier.config import get_settings

    settings = get_settings()
    settings.inference_backend = args.backend
    predictor = SentimentPredictor(settings=settings)

    print(f"Running inference using backend: {args.backend} on {len(texts)} samples...")
    predictions = predictor.predict_batch(texts, languages)
    y_pred = [LABEL_TO_ID[p.label] for p in predictions]

    metrics = calculate_all_metrics(y_true, y_pred)
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted-F1: {metrics['weighted_f1'] * 100:.2f}%")

    save_json(metrics, args.metrics_out)
    print(f"Metrics saved to {args.metrics_out}")

    baseline_metrics = (
        load_json("artifacts/reports/baseline_metrics.json")
        if Path("artifacts/reports/baseline_metrics.json").exists()
        else None
    )
    md_report = generate_markdown_evaluation_report(
        baseline_metrics=baseline_metrics,
        transformer_metrics=metrics if args.backend != "baseline" else None,
        onnx_metrics=metrics if args.backend in ["onnx", "quantized_onnx"] else None,
    )
    save_markdown_report(md_report, args.report_out)
    print(f"Markdown report saved to {args.report_out}")


if __name__ == "__main__":
    main()
