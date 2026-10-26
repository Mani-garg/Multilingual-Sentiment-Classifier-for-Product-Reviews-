"""Text normalization for baseline linear models versus transformer models."""

import re
import unicodedata


def normalize_for_baseline(text: str) -> str:
    """Normalize text for TF-IDF baseline vectorization.

    Keeps punctuation and negation tokens, but normalizes casing and whitespace.
    Does NOT use English-only stopword filtering, because this would corrupt
    non-English words and discard essential negation markers (like 'not', 'no', 'ne...pas', 'nicht').
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text)

    # Convert to lowercase
    text = text.lower()

    # Separate common punctuation with spaces so n-grams can capture sentiment punctuation (! and ?)
    text = re.sub(r"([!?,.:;])", r" \1 ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
