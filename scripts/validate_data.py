"""Standalone dataset validation script."""

import argparse

from sentiment_classifier.data.loading import load_dataset
from sentiment_classifier.data.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset quality against domain schema.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input jsonl/csv/parquet file."
    )
    parser.add_argument(
        "--report-path", type=str, default="artifacts/reports/data_quality_report.json"
    )
    args = parser.parse_args()

    print(f"Validating dataset: {args.input}")
    df = load_dataset(args.input)
    clean_df, report = validate_dataset(df)

    print(
        f"Validation finished. Total rows: {report.rows_read}, Accepted: {report.rows_accepted}, Rejected: {report.rows_rejected}"
    )
    if report.rejection_reasons:
        print("Rejection reasons:")
        for reason, count in report.rejection_reasons.items():
            print(f"  - {reason}: {count}")

    report.save(args.report_path)
    print(f"Quality report written to: {args.report_path}")


if __name__ == "__main__":
    main()
