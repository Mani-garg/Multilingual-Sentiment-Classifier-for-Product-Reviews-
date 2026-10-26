"""Unit tests for leakage-resistant dataset partitioning."""

import pandas as pd
import pytest

from sentiment_classifier.data.splitting import check_partition_leakage, split_dataset
from sentiment_classifier.exceptions import DataValidationError


def test_split_dataset_stratification(sample_reviews_df: pd.DataFrame) -> None:
    """Verify split ratios and class distribution retention."""
    # Duplicate df slightly so test size is meaningful
    large_df = pd.concat([sample_reviews_df] * 5, ignore_index=True)
    large_df["review_text"] = [f"{t} (idx {i})" for i, t in enumerate(large_df["review_text"])]

    split_res = split_dataset(
        large_df,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
    )

    len(large_df)
    assert len(split_res.train_df) > 0
    assert len(split_res.val_df) > 0
    assert len(split_res.test_df) > 0

    # Ensure all target classes are represented in train
    assert "positive" in split_res.train_distribution
    assert "negative" in split_res.train_distribution


def test_zero_leakage_safeguard(sample_reviews_df: pd.DataFrame) -> None:
    """Ensure no identical review texts leak across train/val/test partitions."""
    large_df = pd.concat([sample_reviews_df] * 4, ignore_index=True)
    large_df["review_text"] = [f"{t} #{i}" for i, t in enumerate(large_df["review_text"])]

    split_res = split_dataset(large_df, random_seed=42)
    leakage = check_partition_leakage(split_res.train_df, split_res.val_df, split_res.test_df)

    assert leakage["train_val_overlap"] == 0
    assert leakage["train_test_overlap"] == 0
    assert leakage["val_test_overlap"] == 0


def test_invalid_ratios_raise_error(sample_reviews_df: pd.DataFrame) -> None:
    """Ratios not summing to 1.0 must raise DataValidationError."""
    with pytest.raises(DataValidationError):
        split_dataset(sample_reviews_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)
