"""Execute diagnostic error analysis across the 10 failure mode categories."""

import argparse

from sentiment_classifier.analysis.error_analysis import ErrorAnalyzer
from sentiment_classifier.config import get_settings
from sentiment_classifier.data.loading import load_dataset
from sentiment_classifier.inference.predictor import SentimentPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze classification errors and linguistic failure modes."
    )
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--backend", type=str, default="baseline")
    parser.add_argument(
        "--output", type=str, default="artifacts/reports/error_analysis_report.json"
    )
    args = parser.parse_args()

    print(f"Loading test split: {args.test_data}")
    df = load_dataset(args.test_data)

    settings = get_settings()
    settings.inference_backend = args.backend
    predictor = SentimentPredictor(settings=settings)

    texts = df["review_text"].tolist()
    rids = df["review_id"].tolist()
    true_labels = df["sentiment_label"].tolist()
    languages = df["language"].tolist() if "language" in df.columns else None

    print(f"Predicting on {len(texts)} test samples with '{args.backend}' backend...")
    predictions = predictor.predict_batch(texts, languages)

    pred_labels = [p.label for p in predictions]
    prob_distributions = [p.scores for p in predictions]

    analyzer = ErrorAnalyzer(max_examples_per_category=5)
    report = analyzer.analyze(
        review_ids=rids,
        texts=texts,
        true_labels=true_labels,
        pred_labels=pred_labels,
        prob_distributions=prob_distributions,
        languages=languages,
    )

    print("\n--- ERROR ANALYSIS SUMMARY ---")
    print(f"Total Test Samples: {report['total_samples']}")
    print(f"Total Errors:       {report['total_errors']} ({report['error_rate_pct']:.1f}%)")
    print("\nCategorized Failure Modes:")
    for cat, count in report["category_distribution"].items():
        pct = (count / max(1, report["total_errors"])) * 100.0
        print(f"  - {cat:<30}: {count} occurrences ({pct:.1f}%)")

    analyzer.save_report(report, args.output)
    print(f"\nDetailed error analysis report written to: {args.output}")


if __name__ == "__main__":
    main()
