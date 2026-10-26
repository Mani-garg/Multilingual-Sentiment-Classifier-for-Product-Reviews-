"""Controlled targeted augmentation experiment on training partition only."""

import argparse
import time

from sentiment_classifier.baseline.train import train_baseline_model
from sentiment_classifier.data.loading import load_dataset
from sentiment_classifier.training.augmentation import TargetedAugmenter
from sentiment_classifier.utils.serialization import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate targeted data augmentation on training split."
    )
    parser.add_argument("--train-data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl")
    parser.add_argument(
        "--output-report", type=str, default="artifacts/reports/augmentation_experiment_report.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading partitions: {args.train_data} and {args.test_data}")
    train_df = load_dataset(args.train_data)
    test_df = load_dataset(args.test_data)

    # 1. Evaluate baseline without augmentation
    print("\n[Condition A] Training standard model without augmentation...")
    time.perf_counter()
    baseline_artifacts = train_baseline_model(
        train_df=train_df,
        test_df=test_df,
        artifact_dir="artifacts/baseline",
        random_seed=args.seed,
    )
    base_metrics = baseline_artifacts.metrics

    # 2. Apply targeted augmentation strictly to train_df
    print(
        "\n[Condition B] Applying targeted negation & synonym augmentation strictly to train partition..."
    )
    augmenter = TargetedAugmenter(max_ratio=0.25, random_seed=args.seed)
    augmented_train_df, aug_stats = augmenter.augment_dataset(train_df)
    print(
        f"Original train size: {aug_stats.original_samples} -> Augmented size: {len(augmented_train_df)}"
    )
    print(
        f"Augmented examples: {aug_stats.augmented_samples_created} (Negation: {aug_stats.negation_augmented_count}, Synonym: {aug_stats.synonym_augmented_count})"
    )

    # 3. Train on augmented partition and evaluate on identical held-out test split
    print("Training augmented model...")
    aug_artifacts = train_baseline_model(
        train_df=augmented_train_df,
        test_df=test_df,
        artifact_dir="artifacts/baseline_augmented",
        random_seed=args.seed,
    )
    aug_metrics = aug_artifacts.metrics

    # 4. Compare results
    delta_macro_f1_pts = (aug_metrics["macro_f1"] - base_metrics["macro_f1"]) * 100.0
    delta_acc_pts = (aug_metrics["accuracy"] - base_metrics["accuracy"]) * 100.0

    print("\n--- AUGMENTATION ABLATION COMPARISON ---")
    print(f"Unaugmented Macro-F1: {base_metrics['macro_f1'] * 100:.2f}%")
    print(f"Augmented Macro-F1:   {aug_metrics['macro_f1'] * 100:.2f}%")
    print(f"Absolute Difference:  {delta_macro_f1_pts:+.2f} percentage points")
    print(f"Unaugmented Accuracy: {base_metrics['accuracy'] * 100:.2f}%")
    print(f"Augmented Accuracy:   {aug_metrics['accuracy'] * 100:.2f}%")
    print(f"Absolute Difference:  {delta_acc_pts:+.2f} percentage points")

    report = {
        "dataset_train_size_unaugmented": len(train_df),
        "dataset_train_size_augmented": len(augmented_train_df),
        "augmentation_stats": {
            "augmented_samples_created": aug_stats.augmented_samples_created,
            "negation_count": aug_stats.negation_augmented_count,
            "synonym_count": aug_stats.synonym_augmented_count,
            "rejected_count": aug_stats.rejected_samples,
        },
        "unaugmented_metrics": {
            "macro_f1": base_metrics["macro_f1"],
            "accuracy": base_metrics["accuracy"],
            "weighted_f1": base_metrics["weighted_f1"],
            "per_class": base_metrics["per_class"],
        },
        "augmented_metrics": {
            "macro_f1": aug_metrics["macro_f1"],
            "accuracy": aug_metrics["accuracy"],
            "weighted_f1": aug_metrics["weighted_f1"],
            "per_class": aug_metrics["per_class"],
        },
        "comparison": {
            "macro_f1_delta_percentage_points": round(delta_macro_f1_pts, 3),
            "accuracy_delta_percentage_points": round(delta_acc_pts, 3),
            "conclusion": (
                "Augmentation improved generalization"
                if delta_macro_f1_pts > 0
                else "Augmentation did not improve macro-F1 on held-out test split"
            ),
        },
    }

    save_json(report, args.output_report)
    print(f"\nAugmentation report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
