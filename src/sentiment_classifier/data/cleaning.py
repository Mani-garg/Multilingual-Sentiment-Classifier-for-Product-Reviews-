"""Conservative multilingual text cleaning and normalization."""

import re
import unicodedata

import pandas as pd


def clean_text_conservative(text: str) -> str:
    """Conservatively clean multilingual text while preserving all sentiment markers.

    Transformations:
      - Unicode normalization to NFC (combines accents, standardizes non-Latin glyphs).
      - Replaces weird control characters (e.g. \x00, \x08) with empty string.
      - Collapses irregular multi-spaces and excessive repeated tabs while keeping punctuation.
      - Preserves: casing, diacritics, emojis, punctuation (?!), and negation tokens.

    Args:
        text: Raw review string.

    Returns:
        Conservatively cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode to NFC
    normalized = unicodedata.normalize("NFC", text)

    # Remove non-printable control characters except standard whitespace (\n, \t, \r)
    cleaned = "".join(
        ch for ch in normalized if unicodedata.category(ch)[0] != "C" or ch in "\n\t\r"
    )

    # Collapse excessive horizontal spaces/tabs
    cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)

    # Collapse excessive newlines (more than 2 consecutive)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)

    return cleaned.strip()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative cleaning to review_text while retaining original text for auditing."""
    cleaned_df = df.copy()
    if "original_review_text" not in cleaned_df.columns:
        cleaned_df["original_review_text"] = cleaned_df["review_text"]

    cleaned_df["review_text"] = cleaned_df["review_text"].apply(clean_text_conservative)
    return cleaned_df
