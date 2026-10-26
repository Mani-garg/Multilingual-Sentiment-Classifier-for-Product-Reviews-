"""Evaluation routines for transformer models with global and per-language metrics."""

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase

from sentiment_classifier.baseline.train import evaluate_predictions
from sentiment_classifier.preprocessing.tokenization import DynamicPaddingCollator


def evaluate_transformer(
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    labels: list[int],
    languages: list[str] | None = None,
    batch_size: int = 64,
    device: str = "cpu",
    max_length: int = 128,
) -> dict[str, Any]:
    """Evaluate a transformer sequence classifier over batched data with dynamic padding.

    Returns:
        Dictionary containing aggregate metrics, per-class metrics, per-language metrics,
        confusion matrix, and inference latency statistics.
    """
    model.eval()
    model.to(device)

    # Tokenize individual samples without static padding
    dataset_features = []
    for text, label in zip(texts, labels, strict=False):
        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        dataset_features.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": label,
            }
        )

    collator = DynamicPaddingCollator(tokenizer)
    dataloader = DataLoader(
        dataset_features,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    all_preds = []
    all_probs = []
    total_time = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)

            start = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            elapsed = time.perf_counter() - start
            total_time += elapsed

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1).tolist()

            all_preds.extend(preds)
            all_probs.extend(probs.tolist())

    metrics = evaluate_predictions(labels, all_preds, languages=languages)
    metrics["eval_samples_count"] = len(texts)
    metrics["total_inference_time_seconds"] = float(total_time)
    metrics["avg_latency_ms_per_sample"] = float((total_time / max(1, len(texts))) * 1000.0)

    return metrics
