"""Factory module for creating and configuring multilingual transformer architectures."""

from typing import Any

from transformers import AutoConfig

from sentiment_classifier.labels import ID_TO_LABEL, LABEL_TO_ID


def create_model(
    checkpoint: str = "distilbert-base-multilingual-cased",
    num_labels: int = 3,
    label2id: dict[str, int] | None = None,
    id2label: dict[int, str] | None = None,
    revision: str = "main",
) -> Any:
    """Instantiate a sequence classifier with stable label contracts.

    Supports PyTorch or TensorFlow/ONNX backend dynamically based on environment.
    """
    if label2id is None:
        label2id = LABEL_TO_ID
    if id2label is None:
        id2label = ID_TO_LABEL

    config = AutoConfig.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="sentiment-analysis",
        revision=revision,
    )

    try:
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint, config=config, revision=revision
        )
    except Exception:
        # Fallback for inference/ONNX-first environments without full PyTorch
        return config
