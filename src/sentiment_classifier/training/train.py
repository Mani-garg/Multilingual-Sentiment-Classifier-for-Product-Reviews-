"""Fine-tuning training workflow for multilingual sequence classification."""

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from sentiment_classifier.labels import LABEL_TO_ID
from sentiment_classifier.preprocessing.tokenization import DynamicPaddingCollator, get_tokenizer
from sentiment_classifier.training.callbacks import EarlyStopping
from sentiment_classifier.training.checkpointing import save_transformer_checkpoint
from sentiment_classifier.training.evaluation import evaluate_transformer
from sentiment_classifier.training.losses import WeightedCrossEntropyLoss, get_class_weights
from sentiment_classifier.training.model_factory import create_model
from sentiment_classifier.utils.reproducibility import compute_dataset_fingerprint, set_seed

logger = logging.getLogger(__name__)


def prepare_dataset_features(
    texts: list[str], labels: list[int], tokenizer: Any, max_length: int = 128
) -> list[dict[str, Any]]:
    """Tokenize texts without static padding for dynamic micro-batch collating."""
    features = []
    for text, label in zip(texts, labels, strict=False):
        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        features.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": label,
            }
        )
    return features


def train_transformer_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    checkpoint: str = "distilbert-base-multilingual-cased",
    max_sequence_length: int = 128,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    gradient_clip_norm: float = 1.0,
    use_class_weights: bool = True,
    early_stopping_patience: int = 2,
    random_seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train multilingual transformer model end-to-end with dynamic batching and early stopping.

    Returns:
        Dictionary containing final test evaluation metrics, training duration, and artifact paths.
    """
    set_seed(random_seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting transformer fine-tuning on device: %s", device)

    # 1. Load tokenizer and model
    tokenizer = get_tokenizer(checkpoint=checkpoint)
    model = create_model(checkpoint=checkpoint, num_labels=3)
    model.to(device)

    # 2. Extract texts and integer labels
    train_texts = train_df["review_text"].tolist()
    train_labels = [LABEL_TO_ID[lbl] for lbl in train_df["sentiment_label"]]
    val_texts = val_df["review_text"].tolist()
    val_labels = [LABEL_TO_ID[lbl] for lbl in val_df["sentiment_label"]]
    val_languages = val_df["language"].tolist() if "language" in val_df.columns else None
    test_texts = test_df["review_text"].tolist()
    test_labels = [LABEL_TO_ID[lbl] for lbl in test_df["sentiment_label"]]
    test_languages = test_df["language"].tolist() if "language" in test_df.columns else None

    # Compute dataset fingerprint
    fingerprint = compute_dataset_fingerprint(train_texts)

    # 3. Create datasets and dynamic collator
    train_features = prepare_dataset_features(
        train_texts, train_labels, tokenizer, max_sequence_length
    )
    collator = DynamicPaddingCollator(tokenizer)
    train_loader = DataLoader(
        train_features,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # 4. Setup Optimizer, Scheduler, and Loss
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    class_weights_tensor = None
    if use_class_weights:
        class_weights_tensor = get_class_weights(train_labels, num_classes=3).to(device)
    loss_fn = WeightedCrossEntropyLoss(class_weights=class_weights_tensor)

    # 5. Training Loop with Early Stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
    epoch_logs = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        step_count = 0

        for _step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            targets = torch.tensor(batch["labels"], dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            step_count += 1

        avg_train_loss = total_train_loss / max(1, step_count)

        # Validation phase
        val_metrics = evaluate_transformer(
            model=model,
            tokenizer=tokenizer,
            texts=val_texts,
            labels=val_labels,
            languages=val_languages,
            batch_size=eval_batch_size,
            device=device,
            max_length=max_sequence_length,
        )

        val_macro_f1 = val_metrics["macro_f1"]
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Macro-F1: %.4f | Val Acc: %.4f",
            epoch,
            epochs,
            avg_train_loss,
            val_macro_f1,
            val_metrics["accuracy"],
        )

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        # Early stopping step
        early_stopping.step(val_macro_f1, epoch, model)
        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    training_duration = time.perf_counter() - start_time

    # Restore best checkpoint
    early_stopping.restore_best_weights(model)

    # 6. Final Evaluation on Held-out Test Split
    test_metrics = evaluate_transformer(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        labels=test_labels,
        languages=test_languages,
        batch_size=eval_batch_size,
        device=device,
        max_length=max_sequence_length,
    )
    test_metrics["training_duration_seconds"] = float(training_duration)
    test_metrics["epoch_logs"] = epoch_logs
    test_metrics["best_epoch"] = early_stopping.best_epoch

    training_config = {
        "checkpoint": checkpoint,
        "max_sequence_length": max_sequence_length,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "use_class_weights": use_class_weights,
        "random_seed": random_seed,
    }

    # 7. Persist Checkpoint Artifacts
    saved_path = save_transformer_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        training_config=training_config,
        eval_metrics=test_metrics,
        dataset_fingerprint=fingerprint,
    )

    logger.info("Transformer artifacts successfully saved to %s", saved_path)
    return test_metrics
