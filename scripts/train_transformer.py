"""Fine-tune the multilingual transformer sentiment classifier."""

import argparse
from pathlib import Path

from sentiment_classifier.data.loading import load_dataset
from sentiment_classifier.training.train import train_transformer_model
from sentiment_classifier.utils.serialization import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune multilingual transformer model.")
    parser.add_argument("--train-data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="artifacts/transformer")
    parser.add_argument("--checkpoint", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading partitions from {args.train_data}, {args.val_data}, {args.test_data}")
    train_df = load_dataset(args.train_data)
    val_df = load_dataset(args.val_data)
    test_df = load_dataset(args.test_data)

    print(
        f"Starting fine-tuning: Checkpoint={args.checkpoint}, Epochs={args.epochs}, BatchSize={args.batch_size}"
    )
    metrics = train_transformer_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        max_sequence_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
    )

    print("\n--- TRANSFORMER EVALUATION METRICS ---")
    print(f"Test Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Duration:      {metrics['training_duration_seconds']:.2f} s")

    metrics_out = Path("artifacts/reports/transformer_metrics.json")
    save_json(metrics, metrics_out)
    print(f"Metrics saved to {metrics_out}")


if __name__ == "__main__":
    main()
