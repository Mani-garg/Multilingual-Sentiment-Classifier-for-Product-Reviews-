"""Unit tests for dataset schema, validation, and rating-to-sentiment mappings."""

import pandas as pd
import pytest

from sentiment_classifier.data.validation import validate_dataset
from sentiment_classifier.exceptions import DataValidationError
from sentiment_classifier.labels import SentimentLabel, map_rating_to_sentiment, validate_label


def test_map_rating_to_sentiment_standard_boundaries() -> None:
    """Verify star ratings map strictly to canonical sentiments."""
    assert map_rating_to_sentiment(1.0) == SentimentLabel.NEGATIVE
    assert map_rating_to_sentiment(2.0) == SentimentLabel.NEGATIVE
    assert map_rating_to_sentiment(3.0) == SentimentLabel.NEUTRAL
    assert map_rating_to_sentiment(4.0) == SentimentLabel.POSITIVE
    assert map_rating_to_sentiment(5.0) == SentimentLabel.POSITIVE


def test_map_rating_out_of_bounds() -> None:
    """Out-of-range ratings must raise DataValidationError."""
    with pytest.raises(DataValidationError):
        map_rating_to_sentiment(0.5)

    with pytest.raises(DataValidationError):
        map_rating_to_sentiment(6.0)


def test_validate_label_supported() -> None:
    """Valid labels should be normalized and accepted."""
    assert validate_label("POSITIVE") == "positive"
    assert validate_label(" Neutral ") == "neutral"
    assert validate_label("negative") == "negative"


def test_validate_label_unsupported() -> None:
    """Unsupported labels should raise DataValidationError."""
    with pytest.raises(DataValidationError):
        validate_label("mixed_unknown")


def test_validate_dataset_rejects_empty_and_whitespace(sample_reviews_df: pd.DataFrame) -> None:
    """DataFrame validator must reject empty strings and whitespace-only text."""
    dirty_data = sample_reviews_df.copy()
    # Add empty and whitespace-only rows
    dirty_data.loc[len(dirty_data)] = {
        "review_id": "bad_001",
        "review_text": "   \n\t  ",
        "language": "en",
        "rating": 3.0,
        "sentiment_label": "neutral",
        "group_id": "p_bad",
    }
    dirty_data.loc[len(dirty_data)] = {
        "review_id": "bad_002",
        "review_text": "A",  # Less than min_length=2
        "language": "en",
        "rating": 4.0,
        "sentiment_label": "positive",
        "group_id": "p_bad",
    }

    clean_df, report = validate_dataset(dirty_data, min_length=2)
    assert report.rows_rejected == 2
    assert (
        "whitespace_only_text" in report.rejection_reasons
        or "text_length_under_2" in report.rejection_reasons
    )
    assert len(clean_df) == len(sample_reviews_df)
