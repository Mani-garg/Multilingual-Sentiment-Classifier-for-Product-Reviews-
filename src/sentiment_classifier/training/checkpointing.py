"""Checkpointing utilities for saving and reloading transformer artifacts."""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase

from sentiment_classifier.labels import ID_TO_LABEL, LABEL_TO_ID
from sentiment_classifier.utils.serialization import load_json, save_json


def save_transformer_checkpoint(
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    training_config: dict[str, Any],
    eval_metrics: dict[str, Any],
    dataset_fingerprint: str = "",
) -> Path:
    """Save model weights, tokenizer files, label mappings, and training metadata."""
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save HuggingFace model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save comprehensive metadata
    meta = {
        "model_architecture": model.__class__.__name__,
        "checkpoint_base": model.config.name_or_path,
        "label2id": LABEL_TO_ID,
        "id2label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "training_config": training_config,
        "metrics": eval_metrics,
        "dataset_fingerprint": dataset_fingerprint,
        "model_version": "1.0.0",
        "library_versions": {
            "torch": torch.__version__,
        },
    }
    save_json(meta, save_path / "model_metadata.json")
    return save_path


def load_transformer_checkpoint(
    checkpoint_dir: str | Path,
) -> tuple[AutoModelForSequenceClassification, PreTrainedTokenizerBase, dict[str, Any]]:
    """Reload transformer model, tokenizer, and metadata from disk."""
    load_path = Path(checkpoint_dir)
    from transformers import AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(load_path)
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    metadata = (
        load_json(load_path / "model_metadata.json")
        if (load_path / "model_metadata.json").exists()
        else {}
    )

    return model, tokenizer, metadata
