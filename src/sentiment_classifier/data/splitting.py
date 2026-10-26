"""Leakage-resistant train/validation/test partitioning with group and stratification awareness."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from sentiment_classifier.exceptions import DataValidationError


@dataclass
class SplitResult:
    """Container for split partitions and distribution metadata."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_distribution: dict[str, int]
    val_distribution: dict[str, int]
    test_distribution: dict[str, int]
    language_distributions: dict[str, dict[str, int]]


def check_partition_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "review_text",
) -> dict[str, int]:
    """Check for identical text leakage between train, val, and test splits.

    Returns:
        Dictionary reporting overlap counts between partitions.
    """
    train_texts = set(train_df[text_col].dropna())
    val_texts = set(val_df[text_col].dropna())
    test_texts = set(test_df[text_col].dropna())

    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    val_test_overlap = len(val_texts.intersection(test_texts))

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
    }


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    stratify_col: str = "sentiment_label",
    group_col: str | None = None,
) -> SplitResult:
    """Split dataset into train, validation, and test partitions with leakage safeguards.

    Args:
        df: Input DataFrame.
        train_ratio: Proportion of training records.
        val_ratio: Proportion of validation records.
        test_ratio: Proportion of test records.
        random_seed: Seed for deterministic splitting.
        stratify_col: Target column to maintain class balance across splits.
        group_col: Optional column (e.g. product_id or author_id) to avoid group leakage.

    Returns:
        SplitResult dataclass.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-3):
        raise DataValidationError(f"Split ratios must sum to 1.0 (got {total:.3f})")

    if group_col and group_col in df.columns and df[group_col].nunique() > 10:
        # Use StratifiedGroupKFold to respect group clusters while preserving class balance
        n_splits = int(round(1.0 / test_ratio))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        groups = df[group_col].fillna("unknown")

        # Extract test fold
        train_val_idx, test_idx = next(sgkf.split(df, df[stratify_col], groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Split train_val into train and val
        val_relative_ratio = val_ratio / (train_ratio + val_ratio)
        n_val_splits = int(round(1.0 / val_relative_ratio))
        sgkf_val = StratifiedGroupKFold(
            n_splits=n_val_splits, shuffle=True, random_state=random_seed
        )
        train_val_groups = train_val_df[group_col].fillna("unknown")

        train_idx, val_idx = next(
            sgkf_val.split(train_val_df, train_val_df[stratify_col], train_val_groups)
        )
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    else:
        # Standard stratified train/val/test split
        test_size = int(round(len(df) * test_ratio))
        test_size = max(test_size, 3)

        # Use compound stratification only if test_size is comfortably larger than compound classes
        if "language" in df.columns:
            compound_key = df[stratify_col].astype(str) + "_" + df["language"].astype(str)
            min_count = compound_key.value_counts().min()
            num_classes = compound_key.nunique()
            if min_count >= 3 and test_size >= num_classes and (len(df) - test_size) >= num_classes:
                strat_target = compound_key
            else:
                strat_target = df[stratify_col]
        else:
            strat_target = df[stratify_col]

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=strat_target,
        )

        val_relative_size = val_ratio / (train_ratio + val_ratio)
        val_size = int(round(len(train_val_df) * val_relative_size))
        val_size = max(val_size, 3)

        if "language" in train_val_df.columns:
            compound_key_tv = (
                train_val_df[stratify_col].astype(str) + "_" + train_val_df["language"].astype(str)
            )
            min_count_tv = compound_key_tv.value_counts().min()
            num_classes_tv = compound_key_tv.nunique()
            if (
                min_count_tv >= 3
                and val_size >= num_classes_tv
                and (len(train_val_df) - val_size) >= num_classes_tv
            ):
                strat_target_tv = compound_key_tv
            else:
                strat_target_tv = train_val_df[stratify_col]
        else:
            strat_target_tv = train_val_df[stratify_col]

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=random_seed,
            stratify=strat_target_tv,
        )

    # Post-split leakage check: ensure no identical review_text leaked into val or test
    train_texts = set(train_df["review_text"])
    val_df = val_df[~val_df["review_text"].isin(train_texts)].reset_index(drop=True)
    test_df = test_df[~test_df["review_text"].isin(train_texts)].reset_index(drop=True)

    # Re-index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Compute distributions
    train_dist = train_df[stratify_col].value_counts().to_dict()
    val_dist = val_df[stratify_col].value_counts().to_dict()
    test_dist = test_df[stratify_col].value_counts().to_dict()

    lang_dists = {}
    if "language" in df.columns:
        lang_dists["train"] = train_df["language"].value_counts().to_dict()
        lang_dists["val"] = val_df["language"].value_counts().to_dict()
        lang_dists["test"] = test_df["language"].value_counts().to_dict()

    return SplitResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_distribution=train_dist,
        val_distribution=val_dist,
        test_distribution=test_dist,
        language_distributions=lang_dists,
    )
