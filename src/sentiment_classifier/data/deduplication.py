"""Deduplication utilities for removing identical reviews across languages."""

import pandas as pd


def deduplicate_dataset(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate review entries based on specified columns.

    Args:
        df: Input DataFrame.
        subset: Columns to check for duplicates (defaults to ['review_text', 'language']).
        keep: Which duplicate to retain ('first', 'last', False).

    Returns:
        Tuple of (deduplicated_df, duplicates_dropped_count).
    """
    if subset is None:
        subset = ["review_text", "language"]

    initial_count = len(df)
    deduped_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    duplicates_dropped = initial_count - len(deduped_df)

    return deduped_df, duplicates_dropped
