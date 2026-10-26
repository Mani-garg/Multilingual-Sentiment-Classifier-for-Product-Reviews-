"""Unit tests for conservative cleaning and deduplication."""

import pandas as pd

from sentiment_classifier.data.cleaning import clean_text_conservative
from sentiment_classifier.data.deduplication import deduplicate_dataset


def test_clean_text_conservative_preserves_sentiment_markers() -> None:
    """Ensure emojis, diacritics, exclamation marks, and negation are not stripped."""
    raw = "  Nicht gut! Überhaupt nicht... 😡  \t"
    cleaned = clean_text_conservative(raw)

    assert "Nicht gut" in cleaned
    assert "!" in cleaned
    assert "..." in cleaned
    assert "Überhaupt" in cleaned  # Umlaut preserved
    assert "😡" in cleaned  # Emoji preserved
    assert not cleaned.startswith(" ")
    assert not cleaned.endswith(" ")


def test_clean_text_cjk_preservation() -> None:
    """CJK Japanese and Chinese characters must be preserved without corruption."""
    jp_raw = "  非常に良い製品です！ 素晴らしい ✨  "
    jp_clean = clean_text_conservative(jp_raw)
    assert jp_clean == "非常に良い製品です！ 素晴らしい ✨"

    zh_raw = "质量太差，完全不可用。"
    zh_clean = clean_text_conservative(zh_raw)
    assert zh_clean == "质量太差，完全不可用。"


def test_deduplicate_dataset() -> None:
    """Deduplicator must remove duplicate entries and record count."""
    df = pd.DataFrame(
        [
            {"review_id": "1", "review_text": "Great item", "language": "en"},
            {"review_id": "2", "review_text": "Great item", "language": "en"},  # Duplicate
            {"review_id": "3", "review_text": "Great item", "language": "de"},  # Different language
            {"review_id": "4", "review_text": "Bad item", "language": "en"},
        ]
    )

    deduped, dropped = deduplicate_dataset(df, subset=["review_text", "language"])
    assert len(deduped) == 3
    assert dropped == 1
    assert "1" in deduped["review_id"].values
    assert "2" not in deduped["review_id"].values
