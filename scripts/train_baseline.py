"""Train and evaluate the TF-IDF + Logistic Regression baseline."""

import argparse

from sentiment_classifier.baseline.train import train_baseline_model
from sentiment_classifier.data.loading import load_dataset
from sentiment_classifier.utils.serialization import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear TF-IDF + Logistic Regression baseline."
    )
    parser.add_argument("--train-data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/baseline")
    parser.add_argument(
        "--metrics-out", type=str, default="artifacts/reports/baseline_metrics.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading partitions: {args.train_data} and {args.test_data}")
    train_df = load_dataset(args.train_data)
    test_df = load_dataset(args.test_data)

    print(f"Training TF-IDF baseline on {len(train_df)} examples...")
    artifacts = train_baseline_model(
        train_df=train_df,
        test_df=test_df,
        artifact_dir=args.artifact_dir,
        random_seed=args.seed,
    )

    metrics = artifacts.metrics
    print("\n--- BASELINE EVALUATION RESULTS ---")
    print(f"Macro-F1:     {metrics['macro_f1'] * 100:.2f}%")
    print(f"Accuracy:     {metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted-F1:  {metrics['weighted_f1'] * 100:.2f}%")
    print(f"Training Time: {metrics['training_duration_seconds']:.2f} s")
    print(f"Avg Latency:  {metrics['avg_inference_latency_ms']:.2f} ms")

    print("\nPer-Class Breakdown:")
    for cls_name, c_stats in metrics["per_class"].items():
        print(
            f"  {cls_name.capitalize():<9}: F1={c_stats['f1']:.4f} | Prec={c_stats['precision']:.4f} | Rec={c_stats['recall']:.4f}"
        )

    if metrics.get("per_language"):
        print("\nPer-Language Breakdown:")
        for lang, l_stats in metrics["per_language"].items():
            f1_str = (
                f"{l_stats['macro_f1'] * 100:.1f}%" if l_stats["macro_f1"] is not None else "N/A"
            )
            acc_str = (
                f"{l_stats['accuracy'] * 100:.1f}%" if l_stats["accuracy"] is not None else "N/A"
            )
            print(
                f"  {lang.upper()}: Samples={l_stats['sample_count']}, Macro-F1={f1_str}, Acc={acc_str}, Errors={l_stats['error_count']}"
            )

    save_json(metrics, args.metrics_out)
    print(f"\nBaseline metrics written to: {args.metrics_out}")


if __name__ == "__main__":
    main()
