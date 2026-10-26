"""Language metadata resolution, script detection, and coverage auditing."""

from dataclasses import dataclass

from sentiment_classifier.labels import SUPPORTED_LANGUAGES


@dataclass
class LanguageMetadata:
    """Audit record distinguishing verified dataset language from heuristic detection."""

    language_code: str
    is_supported: bool
    source: str  # 'dataset_metadata', 'heuristic_detected', or 'client_supplied'
    confidence: float = 1.0


def detect_or_verify_language(
    text: str,
    supplied_lang: str | None = None,
) -> LanguageMetadata:
    """Verify supplied language code or detect basic script heuristics.

    Heuristic script detection supports distinguishing Latin (en, de, es, fr),
    CJK Hanzi/Kanji (zh, ja), and Hiragana/Katakana (ja).
    """
    if supplied_lang:
        code = supplied_lang.strip().lower()
        return LanguageMetadata(
            language_code=code,
            is_supported=(code in SUPPORTED_LANGUAGES),
            source="supplied",
            confidence=1.0,
        )

    # Simple character heuristic for unannotated texts
    has_japanese = any("\u3040" <= ch <= "\u309f" or "\u30a0" <= ch <= "\u30ff" for ch in text)
    if has_japanese:
        return LanguageMetadata(
            language_code="ja",
            is_supported=True,
            source="heuristic_detected",
            confidence=0.85,
        )

    has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    if has_chinese:
        return LanguageMetadata(
            language_code="zh",
            is_supported=True,
            source="heuristic_detected",
            confidence=0.80,
        )

    # Default fallback to English for Latin-script text without metadata
    return LanguageMetadata(
        language_code="en",
        is_supported=True,
        source="heuristic_detected",
        confidence=0.50,
    )
