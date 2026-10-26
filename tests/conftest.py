"""Shared pytest fixtures for sentiment classifier test suites."""

from pathlib import Path

import pandas as pd
import pytest

from sentiment_classifier.data.loading import save_dataset


@pytest.fixture
def sample_reviews_df() -> pd.DataFrame:
    """Fixture providing representative multilingual test reviews across classes."""
    data = [
        {
            "review_id": "rev_001",
            "review_text": "Excellent quality, extremely fast shipping!",
            "language": "en",
            "rating": 5.0,
            "sentiment_label": "positive",
            "group_id": "prod_1",
        },
        {
            "review_id": "rev_002",
            "review_text": "Horrible product, broke after 2 hours. Waste of money.",
            "language": "en",
            "rating": 1.0,
            "sentiment_label": "negative",
            "group_id": "prod_1",
        },
        {
            "review_id": "rev_003",
            "review_text": "It is an okay item for the price, average.",
            "language": "en",
            "rating": 3.0,
            "sentiment_label": "neutral",
            "group_id": "prod_2",
        },
        {
            "review_id": "rev_004",
            "review_text": "The battery life is not good at all.",
            "language": "en",
            "rating": 2.0,
            "sentiment_label": "negative",
            "group_id": "prod_2",
        },
        {
            "review_id": "rev_005",
            "review_text": "Not bad for the price, quite useful.",
            "language": "en",
            "rating": 4.0,
            "sentiment_label": "positive",
            "group_id": "prod_3",
        },
        {
            "review_id": "rev_006",
            "review_text": "Hervorragende Qualität und super schneller Versand!",
            "language": "de",
            "rating": 5.0,
            "sentiment_label": "positive",
            "group_id": "prod_3",
        },
        {
            "review_id": "rev_007",
            "review_text": "Völliger Schrott! Geht gar nicht.",
            "language": "de",
            "rating": 1.0,
            "sentiment_label": "negative",
            "group_id": "prod_4",
        },
        {
            "review_id": "rev_008",
            "review_text": "Produit d'excellente qualité, très satisfait.",
            "language": "fr",
            "rating": 5.0,
            "sentiment_label": "positive",
            "group_id": "prod_4",
        },
        {
            "review_id": "rev_009",
            "review_text": "Catastrophique, ne fonctionne pas du tout.",
            "language": "fr",
            "rating": 1.0,
            "sentiment_label": "negative",
            "group_id": "prod_5",
        },
        {
            "review_id": "rev_010",
            "review_text": "¡Excelente producto! Llegó muy rápido.",
            "language": "es",
            "rating": 5.0,
            "sentiment_label": "positive",
            "group_id": "prod_5",
        },
        {
            "review_id": "rev_011",
            "review_text": "Muy malo, pésima calidad y materiales débiles.",
            "language": "es",
            "rating": 1.0,
            "sentiment_label": "negative",
            "group_id": "prod_6",
        },
        {
            "review_id": "rev_012",
            "review_text": "画質が非常に美しく、大満足です！",
            "language": "ja",
            "rating": 5.0,
            "sentiment_label": "positive",
            "group_id": "prod_6",
        },
        {
            "review_id": "rev_013",
            "review_text": "初期不良で最悪でした。買わない方がいいです。",
            "language": "ja",
            "rating": 1.0,
            "sentiment_label": "negative",
            "group_id": "prod_7",
        },
        {
            "review_id": "rev_014",
            "review_text": "屏幕显示效果极佳，做工精良！",
            "language": "zh",
            "rating": 5.0,
            "sentiment_label": "positive",
            "group_id": "prod_7",
        },
        {
            "review_id": "rev_015",
            "review_text": "质量太差，完全无法正常使用。",
            "language": "zh",
            "rating": 1.0,
            "sentiment_label": "negative",
            "group_id": "prod_8",
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def temp_workspace(tmp_path: Path, sample_reviews_df: pd.DataFrame) -> Path:
    """Fixture initializing temporary workspace with sample data and artifact directories."""
    data_dir = tmp_path / "data" / "sample"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "onnx").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "reports").mkdir(parents=True, exist_ok=True)

    sample_file = data_dir / "reviews.jsonl"
    save_dataset(sample_reviews_df, sample_file)
    return tmp_path
