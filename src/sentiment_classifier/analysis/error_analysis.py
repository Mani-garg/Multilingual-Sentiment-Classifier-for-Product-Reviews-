"""Diagnostic error-analysis pipeline categorizing misclassifications across linguistic failure modes."""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sentiment_classifier.utils.serialization import save_json

# Linguistic detection patterns for error taxonomy
NEGATION_REGEX = re.compile(
    r"\b(not|never|no|n't|none|neither|hardly|scarcely|barely|"
    r"nicht|nie|kein|keine|keineswegs|"
    r"pas|jamais|aucun|ne|"
    r"no|nunca|jamás|nada)\b",
    re.IGNORECASE,
)

CONTRASTIVE_REGEX = re.compile(
    r"\b(but|however|although|though|yet|despite|nevertheless|"
    r"aber|jedoch|obwohl|doch|"
    r"mais|cependant|pourtant|bien que|"
    r"pero|sin embargo|aunque)\b",
    re.IGNORECASE,
)

SARCASM_REGEX = re.compile(
    r"(\b(great|wonderful|fantastic|amazing|superb)\b.*\b(broke|failed|useless|waste|garbage|never works|trash)\b)|"
    r"(\b(toll|super|klasse)\b.*\b(kaputt|schrott|müll|nutzlos)\b)|"
    r"(\b(génial|merveilleux)\b.*\b(cassé|inutile|déchet)\b)",
    re.IGNORECASE,
)


@dataclass
class ErrorRecord:
    """Detailed metadata for a single misclassified review."""

    review_id: str
    text_preview: str  # Truncated for privacy
    language: str
    true_label: str
    predicted_label: str
    confidence: float
    probability_distribution: dict[str, float]
    primary_category: str
    secondary_cues: list[str] = field(default_factory=list)


def categorize_error(
    text: str,
    true_label: str,
    pred_label: str,
    confidence: float,
    language: str,
) -> tuple[str, list[str]]:
    """Determine the primary failure category and linguistic cues for a misclassified review.

    Categories:
      1. negation_error
      2. sarcasm_or_irony
      3. mixed_sentiment
      4. ambiguous_neutral
      5. short_context_poor
      6. code_switching_mixed_language
      7. domain_product_terminology
      8. typographical_errors
      9. underrepresented_class
      10. general_misclassification
    """
    cues = []
    text_len = len(text.strip())

    # 1. Check short / context-poor
    if text_len < 25:
        cues.append("short_text")
        if true_label == "neutral" or pred_label == "neutral":
            return "short_context_poor", cues

    # 2. Check sarcasm / irony
    if SARCASM_REGEX.search(text) and (true_label == "negative" and pred_label == "positive"):
        cues.append("surface_positivity_with_negative_context")
        return "sarcasm_or_irony", cues

    # 3. Check negation failure
    if NEGATION_REGEX.search(text):
        cues.append("negation_tokens_present")
        # If true is negative but model predicted positive or neutral
        if (true_label == "negative" and pred_label != "negative") or (
            true_label == "positive" and pred_label != "positive"
        ):
            return "negation_error", cues

    # 4. Check mixed / contrastive sentiment
    if CONTRASTIVE_REGEX.search(text):
        cues.append("contrastive_connective_present")
        return "mixed_sentiment", cues

    # 5. Check ambiguous neutral boundary
    if true_label == "neutral" or pred_label == "neutral":
        cues.append("neutral_boundary_confusion")
        if confidence < 0.60:
            cues.append("low_model_confidence")
        return "ambiguous_neutral", cues

    # 6. Check code-switching (Latin words in CJK texts or vice-versa)
    if language in ["ja", "zh"] and re.search(r"\b[a-zA-Z]{3,}\b", text):
        cues.append("latin_tokens_in_cjk")
        return "code_switching_mixed_language", cues

    # 7. Check domain product specs / terminology
    if re.search(
        r"\b(ghz|ram|cpu|usb|firmware|mah|fps|bluetooth|hdmi|resolution)\b", text, re.IGNORECASE
    ):
        cues.append("technical_specs_present")
        return "domain_product_terminology", cues

    # 8. Check typographical anomalies
    if re.search(r"(\w)\1{3,}", text):  # Repeated chars e.g. "sooooo"
        cues.append("repeated_characters")
        return "typographical_errors", cues

    return "general_misclassification", cues


class ErrorAnalyzer:
    """Comprehensive error analysis suite aggregating misclassifications into actionable insights."""

    def __init__(self, max_examples_per_category: int = 10) -> None:
        self.max_examples_per_category = max_examples_per_category

    def analyze(
        self,
        review_ids: list[str],
        texts: list[str],
        true_labels: list[str],
        pred_labels: list[str],
        prob_distributions: list[dict[str, float]],
        languages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute diagnostic error analysis across all misclassifications.

        Returns:
            Structured error analysis report dictionary.
        """
        if languages is None:
            languages = ["en"] * len(texts)

        total_samples = len(texts)
        misclassified_records: list[ErrorRecord] = []
        category_counts: dict[str, int] = {}
        examples_by_category: dict[str, list[dict[str, Any]]] = {}

        for rid, text, true_l, pred_l, probs, lang in zip(
            review_ids, texts, true_labels, pred_labels, prob_distributions, languages, strict=False
        ):
            if true_l == pred_l:
                continue

            conf = float(probs.get(pred_l, 0.0))
            category, cues = categorize_error(text, true_l, pred_l, conf, lang)

            # Sanitize preview text to at most 120 chars
            preview = text[:120].strip() + ("..." if len(text) > 120 else "")

            rec = ErrorRecord(
                review_id=rid,
                text_preview=preview,
                language=lang,
                true_label=true_l,
                predicted_label=pred_l,
                confidence=conf,
                probability_distribution=probs,
                primary_category=category,
                secondary_cues=cues,
            )
            misclassified_records.append(rec)
            category_counts[category] = category_counts.get(category, 0) + 1

            if category not in examples_by_category:
                examples_by_category[category] = []
            if len(examples_by_category[category]) < self.max_examples_per_category:
                examples_by_category[category].append(asdict(rec))

        error_rate = (len(misclassified_records) / max(1, total_samples)) * 100.0

        report = {
            "total_samples": total_samples,
            "total_errors": len(misclassified_records),
            "error_rate_pct": float(round(error_rate, 2)),
            "category_distribution": category_counts,
            "representative_examples": examples_by_category,
        }
        return report

    def save_report(self, report: dict[str, Any], path: str | Path) -> None:
        """Save error analysis report to JSON."""
        save_json(report, path)
