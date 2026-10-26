"""Tokenizer service for inference with truncation tracking and warning generation."""

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenizerService:
    """Encapsulates HuggingFace tokenizer with truncation detection and dynamic padding."""

    def __init__(
        self,
        checkpoint: str = "distilbert-base-multilingual-cased",
        max_length: int = 128,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_length = max_length
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        return self._tokenizer

    def tokenize_batch(self, texts: list[str]) -> tuple[dict[str, np.ndarray], list[list[str]]]:
        """Tokenize batch of texts dynamically padding to the batch maximum.

        Returns:
            Tuple of (tensors_dict, per_sample_warnings).
        """
        tok = self.tokenizer
        warnings: list[list[str]] = [[] for _ in texts]

        # Check for length truncation warnings
        for i, text in enumerate(texts):
            raw_tokens = tok.encode(text, add_special_tokens=False)
            if len(raw_tokens) > (self.max_length - 2):  # Accounting for [CLS] and [SEP]
                warnings[i].append(
                    f"Input review exceeded max sequence length ({len(raw_tokens)} > {self.max_length}); truncated."
                )

        # Batch encode with dynamic padding up to batch max
        encoded = tok(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="np",
        )

        input_dict = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        return input_dict, warnings
