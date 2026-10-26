"""Baseline prediction interface for single and batched inference."""

from typing import Any

import numpy as np

from sentiment_classifier.baseline.artifacts import BaselineArtifacts
from sentiment_classifier.preprocessing.text_normalization import normalize_for_baseline


class BaselinePredictor:
    """Predictor wrapping fitted TF-IDF Vectorizer and Logistic Regression classifier."""

    def __init__(self, artifacts: BaselineArtifacts) -> None:
        self.artifacts = artifacts
        self.vectorizer = artifacts.vectorizer
        self.classifier = artifacts.classifier
        self.id_to_label = artifacts.id_to_label
        self.classes_ = list(self.classifier.classes_)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment label and class probability distribution for input texts.

        Returns:
            List of dicts with 'label' and 'scores' mapping.
        """
        if not texts:
            return []

        # Normalize texts conservatively
        norm_texts = [normalize_for_baseline(t) for t in texts]

        # Vectorize
        x_tfidf = self.vectorizer.transform(norm_texts)

        # Predict probabilities
        probs = self.classifier.predict_proba(x_tfidf)
        preds = np.argmax(probs, axis=1)

        results = []
        for pred_idx, prob_row in zip(preds, probs, strict=False):
            class_id = self.classes_[pred_idx]
            label = self.id_to_label.get(class_id, "unknown")
            score_dict = {
                self.id_to_label.get(c, str(c)): float(prob_row[i])
                for i, c in enumerate(self.classes_)
            }
            results.append(
                {
                    "label": label,
                    "scores": score_dict,
                    "model_version": "baseline-tfidf-logreg",
                    "inference_backend": "baseline",
                }
            )

        return results
