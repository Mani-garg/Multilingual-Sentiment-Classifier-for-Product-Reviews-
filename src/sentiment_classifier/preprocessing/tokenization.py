"""Tokenization helpers and dynamic padding collators for multilingual transformers."""

from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer(
    checkpoint: str = "distilbert-base-multilingual-cased",
    use_fast: bool = True,
    do_lower_case: bool = False,
) -> PreTrainedTokenizerBase:
    """Load pretrained HuggingFace tokenizer with multilingual settings."""
    return AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=use_fast,
        do_lower_case=do_lower_case,
    )


def tokenize_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    padding: bool | str = False,
    truncation: bool = True,
    return_tensors: str | None = None,
) -> dict[str, Any]:
    """Tokenize a sequence of multilingual texts with truncation policies."""
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )


class DynamicPaddingCollator:
    """Batch collator that pads tensors to the longest sequence in the micro-batch.

    This prevents wasteful padding to the global max_sequence_length and
    significantly accelerates transformer forward/backward passes.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """Collate list of feature dicts (each having 'input_ids' and optionally 'attention_mask' or 'label')."""
        batch = {}
        first = features[0]

        # Determine batch max length for sequence fields
        seq_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 0:
            seq_len = (
                (seq_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Pad input_ids
        batch_input_ids = np.full((len(features), seq_len), pad_id, dtype=np.int64)
        for i, f in enumerate(features):
            ids = f["input_ids"]
            batch_input_ids[i, : len(ids)] = ids
        batch["input_ids"] = batch_input_ids

        # Pad attention_mask
        if "attention_mask" in first:
            batch_mask = np.zeros((len(features), seq_len), dtype=np.int64)
            for i, f in enumerate(features):
                mask = f["attention_mask"]
                batch_mask[i, : len(mask)] = mask
            batch["attention_mask"] = batch_mask

        # Labels (if present)
        if "label" in first or "labels" in first:
            key = "label" if "label" in first else "labels"
            batch["labels"] = np.array([f[key] for f in features], dtype=np.int64)

        return batch
