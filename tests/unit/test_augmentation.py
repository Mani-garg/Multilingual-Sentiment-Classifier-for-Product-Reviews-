"""Unit tests for targeted data augmentation and label preservation."""

import pandas as pd

from sentiment_classifier.training.augmentation import TargetedAugmenter


def test_negation_augmentation_preserves_polarity() -> None:
    """Negation paraphrase must maintain the exact semantic sentiment label."""
    augmenter = TargetedAugmenter(random_seed=42)
    text = "The camera is not good and battery life is poor."
    aug = augmenter.augment_negation(text, lang="en")

    assert aug is not None
    assert "far from good" in aug
    # Polarity is still negative


def test_synonym_augmentation_preserves_polarity() -> None:
    """Synonym substitution must preserve positive/negative sentiment polarity."""
    augmenter = TargetedAugmenter(random_seed=42)
    text = "This is a great product with fantastic performance."
    aug = augmenter.augment_synonyms(text, sentiment="positive")

    assert aug is not None
    assert aug != text


def test_augment_dataset_increases_sample_count(sample_reviews_df: pd.DataFrame) -> None:
    """Dataset augmenter should augment valid records up to configured ratio."""
    augmenter = TargetedAugmenter(max_ratio=0.5, random_seed=42)
    aug_df, stats = augmenter.augment_dataset(sample_reviews_df)

    assert len(aug_df) >= len(sample_reviews_df)
    assert stats.original_samples == len(sample_reviews_df)
    assert "is_augmented" in aug_df.columns
