"""Training callbacks for early stopping and metric tracking."""

import copy
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping monitor that halts training when validation metric ceases improving."""

    def __init__(
        self,
        patience: int = 2,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.best_weights: dict[str, Any] | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        return score < (self.best_score - self.min_delta)

    def step(self, score: float, epoch: int, model: torch.nn.Module) -> bool:
        """Update monitor with current epoch score. Returns True if score improved."""
        if self.is_better(score):
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True

        self.counter += 1
        logger.info("Early stopping counter: %d / %d", self.counter, self.patience)
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        """Restore the best model weights observed during training."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(
                "Restored best model weights from epoch %d (score: %.4f)",
                self.best_epoch,
                self.best_score or 0.0,
            )
