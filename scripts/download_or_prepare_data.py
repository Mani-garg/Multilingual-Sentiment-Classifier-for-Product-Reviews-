"""Ingestion and preparation script: validation, cleaning, deduplication, and splitting."""

import argparse
from pathlib import Path

from sentiment_classifier.data.cleaning import clean_dataset
from sentiment_classifier.data.deduplication import deduplicate_dataset
from sentiment_classifier.data.loading import load_dataset, save_dataset
from sentiment_classifier.data.provenance import DatasetProvenance
from sentiment_classifier.data.splitting import check_partition_leakage, split_dataset
from sentiment_classifier.data.validation import validate_dataset
from sentiment_classifier.utils.reproducibility import compute_dataset_fingerprint, set_seed


def prepare_data_pipeline(
    raw_path: str | Path,
    processed_dir: str | Path,
    report_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    set_seed(seed)
    p_dir = Path(processed_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {raw_path}")
    raw_df = load_dataset(raw_path)
    print(f"Raw records loaded: {len(raw_df)}")

    # 1. Validation
    valid_df, quality_report = validate_dataset(raw_df)
    print(f"Validated records: {len(valid_df)} (Rejected: {quality_report.rows_rejected})")

    # 2. Conservative Cleaning
    cleaned_df = clean_dataset(valid_df)

    # 3. Deduplication
    deduped_df, dropped_dupes = deduplicate_dataset(cleaned_df)
    quality_report.duplicate_rows_dropped = dropped_dupes
    print(f"Deduplicated records: {len(deduped_df)} (Duplicates dropped: {dropped_dupes})")

    # 4. Leakage-Resistant Splitting
    split_res = split_dataset(
        deduped_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )

    leakage_check = check_partition_leakage(split_res.train_df, split_res.val_df, split_res.test_df)
    print(f"Partition Leakage Checks: {leakage_check}")
    assert leakage_check["train_val_overlap"] == 0, "Train-Val text leakage detected!"
    assert leakage_check["train_test_overlap"] == 0, "Train-Test text leakage detected!"

    # 5. Persist Partitions
    save_dataset(split_res.train_df, p_dir / "train.jsonl")
    save_dataset(split_res.val_df, p_dir / "val.jsonl")
    save_dataset(split_res.test_df, p_dir / "test.jsonl")

    print(
        f"Saved partitions -> {p_dir}/ (Train: {len(split_res.train_df)}, Val: {len(split_res.val_df)}, Test: {len(split_res.test_df)})"
    )

    # 6. Save Reports & Provenance
    if report_path:
        quality_report.save(report_path)
        print(f"Data quality report saved -> {report_path}")

    if provenance_path:
        fingerprint = compute_dataset_fingerprint(split_res.train_df["review_text"].tolist())
        provenance = DatasetProvenance(
            total_records=len(raw_df),
            clean_records=len(deduped_df),
            duplicate_records_removed=dropped_dupes,
            languages=sorted(deduped_df["language"].unique()),
            checksum_sha256=fingerprint,
        )
        provenance.save(provenance_path)
        print(f"Dataset provenance record saved -> {provenance_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, validate, and split multilingual dataset."
    )
    parser.add_argument(
        "--input", type=str, default="data/sample/multilingual_reviews_sample.jsonl"
    )
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--report", type=str, default="artifacts/reports/data_quality_report.json")
    parser.add_argument(
        "--provenance", type=str, default="artifacts/reports/dataset_provenance.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_data_pipeline(
        raw_path=args.input,
        processed_dir=args.output_dir,
        report_path=args.report,
        provenance_path=args.provenance,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
