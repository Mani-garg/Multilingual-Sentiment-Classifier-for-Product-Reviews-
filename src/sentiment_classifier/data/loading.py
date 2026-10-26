"""Dataset loading and persistence abstractions supporting JSONL, Parquet, and CSV."""

import uuid
from pathlib import Path

import pandas as pd

from sentiment_classifier.exceptions import DataValidationError
from sentiment_classifier.labels import map_rating_to_sentiment, validate_label


def normalize_schema(df: pd.DataFrame, source_id: str = "dataset") -> pd.DataFrame:
    """Normalize arbitrary review datasets into canonical schema.

    Canonical columns:
      - review_id (str)
      - review_text (str)
      - language (str)
      - rating (optional float)
      - sentiment_label (str: negative | neutral | positive)
      - source_dataset (str)
      - source_record_id (optional str)
      - group_id (optional str)
    """
    normalized = df.copy()

    # Column aliases mapping
    aliases = {
        "text": "review_text",
        "body": "review_text",
        "content": "review_text",
        "review_body": "review_text",
        "id": "review_id",
        "lang": "language",
        "stars": "rating",
        "score": "rating",
        "star_rating": "rating",
        "label": "sentiment_label",
        "sentiment": "sentiment_label",
        "product_id": "group_id",
        "asin": "group_id",
    }
    for old_col, new_col in aliases.items():
        if old_col in normalized.columns and new_col not in normalized.columns:
            normalized[new_col] = normalized[old_col]

    if "review_text" not in normalized.columns:
        raise DataValidationError("Input dataset missing 'review_text' column.")

    # Assign stable review_id if missing
    if "review_id" not in normalized.columns:
        normalized["review_id"] = [str(uuid.uuid4()) for _ in range(len(normalized))]
    else:
        normalized["review_id"] = normalized["review_id"].astype(str)

    # Assign language if missing
    if "language" not in normalized.columns:
        normalized["language"] = "en"
    else:
        normalized["language"] = normalized["language"].astype(str).str.lower().str.strip()

    # Assign source_dataset
    if "source_dataset" not in normalized.columns:
        normalized["source_dataset"] = source_id

    # Derive sentiment_label if missing but rating is present
    if "sentiment_label" not in normalized.columns or normalized["sentiment_label"].isna().all():
        if "rating" in normalized.columns and not normalized["rating"].isna().all():
            normalized["sentiment_label"] = normalized["rating"].apply(
                lambda r: map_rating_to_sentiment(r).value if pd.notna(r) else None
            )
        else:
            raise DataValidationError(
                "Dataset must contain either 'sentiment_label' or numerical 'rating' to infer sentiment."
            )
    else:
        normalized["sentiment_label"] = normalized["sentiment_label"].apply(
            lambda val: validate_label(str(val)) if pd.notna(val) else None
        )

    return normalized


def load_dataset(path: str | Path, source_id: str = "dataset") -> pd.DataFrame:
    """Load dataset from disk based on file extension."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".jsonl" or suffix == ".json":
        df = pd.read_json(file_path, lines=(suffix == ".jsonl"), encoding="utf-8")
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix in [".csv", ".tsv"]:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return normalize_schema(df, source_id=source_id)


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe to disk in appropriate format (JSONL or Parquet)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        df.to_json(target, orient="records", lines=True, force_ascii=False)
    elif suffix == ".parquet":
        df.to_parquet(target, index=False)
    elif suffix == ".csv":
        df.to_csv(target, index=False, encoding="utf-8")
    else:
        # Default to JSONL
        df.to_json(target, orient="records", lines=True, force_ascii=False)
