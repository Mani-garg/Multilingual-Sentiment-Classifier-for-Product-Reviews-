"""Loss functions and class-weight computation for sentiment imbalance."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(
    train_labels: list[int],
    num_classes: int = 3,
) -> torch.Tensor:
    """Compute balanced class weights exclusively from training labels."""
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )
    return torch.tensor(weights, dtype=torch.float32)


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with support for class weighting."""

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)
