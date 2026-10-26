"""Unit tests for diagnostic linguistic error categorization."""

from sentiment_classifier.analysis.error_analysis import categorize_error


def test_categorize_sarcasm_error() -> None:
    """Surface positive review with negative rating should trigger sarcasm categorization."""
    text = "Oh great, it broke down on the very first day. Amazing garbage."
    cat, cues = categorize_error(
        text=text,
        true_label="negative",
        pred_label="positive",
        confidence=0.91,
        language="en",
    )
    assert cat == "sarcasm_or_irony"
    assert "surface_positivity_with_negative_context" in cues


def test_categorize_negation_error() -> None:
    """Misclassified negation should be identified as negation_error."""
    text = "The headphones are not good and I never recommend them."
    cat, cues = categorize_error(
        text=text,
        true_label="negative",
        pred_label="positive",
        confidence=0.82,
        language="en",
    )
    assert cat == "negation_error"
    assert "negation_tokens_present" in cues


def test_categorize_short_text_error() -> None:
    """Very short reviews should be identified as short_context_poor."""
    text = "Ok."
    cat, cues = categorize_error(
        text=text,
        true_label="neutral",
        pred_label="positive",
        confidence=0.55,
        language="en",
    )
    assert cat == "short_context_poor"
    assert "short_text" in cues
