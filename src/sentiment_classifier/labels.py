"""Label definitions, enum contracts, and rating-to-sentiment mappings."""

from enum import StrEnum

from sentiment_classifier.exceptions import DataValidationError


class SentimentLabel(StrEnum):
    """Canonical three-class sentiment categories."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


LABEL_TO_ID: dict[str, int] = {
    SentimentLabel.NEGATIVE.value: 0,
    SentimentLabel.NEUTRAL.value: 1,
    SentimentLabel.POSITIVE.value: 2,
}

ID_TO_LABEL: dict[int, str] = {
    0: SentimentLabel.NEGATIVE.value,
    1: SentimentLabel.NEUTRAL.value,
    2: SentimentLabel.POSITIVE.value,
}

SUPPORTED_LABELS: list[str] = [
    SentimentLabel.NEGATIVE.value,
    SentimentLabel.NEUTRAL.value,
    SentimentLabel.POSITIVE.value,
]

# Supported language codes verified against multilingual DistilBERT coverage
SUPPORTED_LANGUAGES: set[str] = {"en", "de", "es", "fr", "ja", "zh", "it", "nl", "pt", "ru"}


def map_rating_to_sentiment(
    rating: float | int,
    negative_threshold: float = 2.0,
    positive_threshold: float = 4.0,
) -> SentimentLabel:
    """Map numerical star ratings (standard 1-5 scale) to canonical sentiment classes.

    Mapping policy:
      - rating <= negative_threshold (e.g. 1, 2) -> NEGATIVE
      - negative_threshold < rating < positive_threshold (e.g. 3) -> NEUTRAL
      - rating >= positive_threshold (e.g. 4, 5) -> POSITIVE

    Args:
        rating: Numerical review rating.
        negative_threshold: Upper inclusive bound for negative class.
        positive_threshold: Lower inclusive bound for positive class.

    Returns:
        SentimentLabel enum value.

    Raises:
        DataValidationError: If rating is outside valid [1.0, 5.0] range.
    """
    if rating < 1.0 or rating > 5.0:
        raise DataValidationError(
            f"Invalid rating: {rating}. Must be within [1.0, 5.0].",
            details={"rating": rating},
        )

    if rating <= negative_threshold:
        return SentimentLabel.NEGATIVE
    if rating >= positive_threshold:
        return SentimentLabel.POSITIVE
    return SentimentLabel.NEUTRAL


def validate_label(label: str) -> str:
    """Validate that a label string belongs to supported sentiment classes."""
    norm = label.strip().lower()
    if norm not in LABEL_TO_ID:
        raise DataValidationError(
            f"Unsupported sentiment label: '{label}'. Must be one of {SUPPORTED_LABELS}",
            details={"received": label, "allowed": SUPPORTED_LABELS},
        )
    return norm
