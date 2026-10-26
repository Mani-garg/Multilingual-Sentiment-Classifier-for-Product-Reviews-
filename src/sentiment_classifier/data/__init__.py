"""Data ingestion, validation, conservative cleaning, deduplication, and splitting modules."""

from sentiment_classifier.data.cleaning import clean_dataset
from sentiment_classifier.data.deduplication import deduplicate_dataset
from sentiment_classifier.data.loading import load_dataset, save_dataset
from sentiment_classifier.data.splitting import SplitResult, split_dataset
from sentiment_classifier.data.validation import DataQualityReport, validate_dataset

__all__ = [
    "load_dataset",
    "save_dataset",
    "validate_dataset",
    "DataQualityReport",
    "clean_dataset",
    "deduplicate_dataset",
    "split_dataset",
    "SplitResult",
]
