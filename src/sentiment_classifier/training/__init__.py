"""Transformer model factory, loss functions, and augmentation."""

from sentiment_classifier.training.augmentation import TargetedAugmenter
from sentiment_classifier.training.model_factory import create_model


def train_transformer_model(*args, **kwargs):
    from sentiment_classifier.training.train import train_transformer_model as _train

    return _train(*args, **kwargs)


__all__ = [
    "create_model",
    "TargetedAugmenter",
    "train_transformer_model",
]
