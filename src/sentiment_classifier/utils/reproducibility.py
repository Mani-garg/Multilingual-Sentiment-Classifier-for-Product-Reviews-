"""Reproducibility utilities for setting deterministic execution seeds across frameworks."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python random, NumPy, and TensorFlow (if available).

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Enable deterministic ops where supported
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def compute_dataset_fingerprint(texts: list[str]) -> str:
    """Compute a deterministic SHA-256 fingerprint from an ordered sequence of texts."""
    import hashlib

    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8", errors="replace"))
        hasher.update(b"\n---RECORD_SEP---\n")
    return hasher.hexdigest()
