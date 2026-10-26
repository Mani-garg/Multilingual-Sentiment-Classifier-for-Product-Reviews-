"""Validation module producing actionable data quality reports."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from sentiment_classifier.labels import SUPPORTED_LABELS, SUPPORTED_LANGUAGES
from sentiment_classifier.utils.serialization import save_json


@dataclass
class DataQualityReport:
    """Detailed summary of data quality metrics, rejections, and invariants."""

    rows_read: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0
    duplicate_rows_dropped: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    language_distribution: dict[str, int] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    text_length_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)


def validate_dataset(
    df: pd.DataFrame,
    min_length: int = 2,
    max_length: int = 5000,
    check_languages: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate dataframe rows against schema invariants and quality constraints.

    Args:
        df: Input pandas DataFrame.
        min_length: Minimum character length for review_text.
        max_length: Maximum character length for review_text.
        check_languages: If True, check language code against known supported set.

    Returns:
        Tuple of (valid_dataframe, data_quality_report).
    """
    report = DataQualityReport(rows_read=len(df))
    rejection_counts: dict[str, int] = {}

    def log_rejection(reason: str, count: int) -> None:
        if count > 0:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + count

    # 1. Required column check
    required_cols = ["review_id", "review_text", "language", "sentiment_label"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    working_df = df.copy()

    # 2. Check nulls / empty values in required fields
    null_text_mask = working_df["review_text"].isna()
    log_rejection("missing_review_text", int(null_text_mask.sum()))

    null_label_mask = working_df["sentiment_label"].isna()
    log_rejection("missing_sentiment_label", int(null_label_mask.sum()))

    null_lang_mask = working_df["language"].isna()
    log_rejection("missing_language", int(null_lang_mask.sum()))

    valid_mask = (~null_text_mask) & (~null_label_mask) & (~null_lang_mask)
    working_df = working_df[valid_mask].copy()

    # 3. Ensure string types
    working_df["review_text"] = working_df["review_text"].astype(str)
    working_df["sentiment_label"] = (
        working_df["sentiment_label"].astype(str).str.strip().str.lower()
    )
    working_df["language"] = working_df["language"].astype(str).str.strip().str.lower()

    # 4. Whitespace-only or empty strings
    empty_ws_mask = working_df["review_text"].str.strip().str.len() == 0
    log_rejection("whitespace_only_text", int(empty_ws_mask.sum()))
    working_df = working_df[~empty_ws_mask].copy()

    # 5. Text length bounds
    text_lengths = working_df["review_text"].str.len()
    too_short_mask = text_lengths < min_length
    log_rejection(f"text_length_under_{min_length}", int(too_short_mask.sum()))

    too_long_mask = text_lengths > max_length
    log_rejection(f"text_length_over_{max_length}", int(too_long_mask.sum()))

    working_df = working_df[(~too_short_mask) & (~too_long_mask)].copy()

    # 6. Valid sentiment labels
    invalid_label_mask = ~working_df["sentiment_label"].isin(SUPPORTED_LABELS)
    log_rejection("invalid_sentiment_label", int(invalid_label_mask.sum()))
    working_df = working_df[~invalid_label_mask].copy()

    # 7. Language code validation
    if check_languages:
        unsupported_lang_mask = ~working_df["language"].isin(SUPPORTED_LANGUAGES)
        log_rejection("unsupported_language_code", int(unsupported_lang_mask.sum()))
        working_df = working_df[~unsupported_lang_mask].copy()

    # 8. Rating consistency (if rating column is present)
    if "rating" in working_df.columns:
        invalid_rating_mask = (working_df["rating"] < 1.0) | (working_df["rating"] > 5.0)
        log_rejection("invalid_rating_range", int(invalid_rating_mask.sum()))
        working_df = working_df[~invalid_rating_mask].copy()

    # Populate summary statistics
    report.rows_accepted = len(working_df)
    report.rows_rejected = report.rows_read - report.rows_accepted
    report.rejection_reasons = rejection_counts
    report.language_distribution = working_df["language"].value_counts().to_dict()
    report.label_distribution = working_df["sentiment_label"].value_counts().to_dict()

    if len(working_df) > 0:
        lengths = working_df["review_text"].str.len()
        report.text_length_stats = {
            "min": float(lengths.min()),
            "max": float(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
        }

    return working_df.reset_index(drop=True), report
