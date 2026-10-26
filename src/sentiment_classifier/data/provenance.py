"""Dataset provenance metadata tracking and documentation generation."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sentiment_classifier.utils.serialization import save_json


@dataclass
class DatasetProvenance:
    """Detailed provenance metadata record for reproducible dataset auditing."""

    dataset_name: str = "Multilingual Amazon Reviews Corpus"
    citation: str = (
        "Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). "
        "The Multilingual Amazon Reviews Corpus. In Proceedings of EMNLP 2020."
    )
    license: str = "Amazon Academic License (Free for academic and research evaluation)"
    source_url: str = "https://huggingface.co/datasets/amazon_reviews_multi"
    version: str = "1.0.0"
    languages: list[str] = field(default_factory=lambda: ["en", "de", "es", "fr", "ja", "zh"])
    total_records: int = 0
    clean_records: int = 0
    duplicate_records_removed: int = 0
    label_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "1_star": "negative",
            "2_star": "negative",
            "3_star": "neutral",
            "4_star": "positive",
            "5_star": "positive",
        }
    )
    missing_value_policy: str = "Drop rows where review_text or sentiment_label is missing/empty"
    duplicate_policy: str = (
        "Deduplicate on exact (review_text, language) match, retaining first occurrence"
    )
    known_biases: list[str] = field(
        default_factory=lambda: [
            "Product domain vocabulary skew towards electronics and household consumer goods",
            "Higher proportion of 5-star positive reviews compared to neutral 3-star reviews",
            "Cultural differences in review length and phrasing across languages (e.g. Japanese reviews tend to be more polite and concise)",
        ]
    )
    checksum_sha256: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)
