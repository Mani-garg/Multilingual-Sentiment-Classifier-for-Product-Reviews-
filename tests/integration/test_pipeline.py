"""Integration test verifying end-to-end data preparation, baseline training, and prediction."""

from pathlib import Path

import pandas as pd
import pytest

from sentiment_classifier.baseline.artifacts import BaselineArtifacts
from sentiment_classifier.baseline.predict import BaselinePredictor
from sentiment_classifier.baseline.train import train_baseline_model
from sentiment_classifier.data.cleaning import clean_dataset
from sentiment_classifier.data.deduplication import deduplicate_dataset
from sentiment_classifier.data.splitting import split_dataset
from sentiment_classifier.data.validation import validate_dataset


def test_full_baseline_pipeline_integration(
    tmp_path: Path, sample_reviews_df: pd.DataFrame
) -> None:
    """Execute validation -> cleaning -> splitting -> baseline training -> inference."""
    # 1. Validation & Cleaning
    valid_df, _ = validate_dataset(sample_reviews_df)
    clean_df = clean_dataset(valid_df)
    dedup_df, _ = deduplicate_dataset(clean_df)

    # 2. Splitting (duplicate slightly so splits have >= 1 item each)
    expanded_df = pd.concat([dedup_df] * 3, ignore_index=True)
    expanded_df["review_text"] = [f"{t} ({i})" for i, t in enumerate(expanded_df["review_text"])]
    split_res = split_dataset(
        expanded_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    # 3. Train Baseline Model
    artifact_dir = tmp_path / "baseline_model"
    artifacts = train_baseline_model(
        train_df=split_res.train_df,
        test_df=split_res.test_df,
        artifact_dir=artifact_dir,
    )

    assert (artifact_dir / "vectorizer.joblib").exists()
    assert (artifact_dir / "classifier.joblib").exists()
    assert (artifact_dir / "metadata.json").exists()
    assert "macro_f1" in artifacts.metrics

    # 4. Reload Artifacts and Run Predictor
    reloaded_artifacts = BaselineArtifacts.load(artifact_dir)
    predictor = BaselinePredictor(reloaded_artifacts)

    test_samples = [
        "This is an amazing, top-quality device!",
        "Völliger Schrott, ging direkt kaputt.",
    ]
    results = predictor.predict(test_samples)

    assert len(results) == 2
    for r in results:
        assert "label" in r
        assert "scores" in r
        assert r["label"] in ["positive", "neutral", "negative"]
        assert len(r["scores"]) == 3
        # Scores sum to ~1.0
        assert pytest.approx(sum(r["scores"].values()), abs=1e-3) == 1.0
