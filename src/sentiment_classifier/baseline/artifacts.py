"""Baseline artifact serialisation and reloading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sentiment_classifier.utils.serialization import load_joblib, load_json, save_joblib, save_json


@dataclass
class BaselineArtifacts:
    """Container for fitted baseline components and metadata."""

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    config: dict[str, Any]
    metrics: dict[str, Any]

    def save(self, artifact_dir: str | Path) -> None:
        """Save vectorizer, classifier, and metadata to artifact_dir."""
        target = Path(artifact_dir)
        target.mkdir(parents=True, exist_ok=True)

        save_joblib(self.vectorizer, target / "vectorizer.joblib")
        save_joblib(self.classifier, target / "classifier.joblib")
        save_json(
            {
                "label_to_id": self.label_to_id,
                "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
                "config": self.config,
                "metrics": self.metrics,
            },
            target / "metadata.json",
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "BaselineArtifacts":
        """Load baseline artifacts from artifact_dir."""
        target = Path(artifact_dir)
        vec = load_joblib(target / "vectorizer.joblib")
        clf = load_joblib(target / "classifier.joblib")
        meta = load_json(target / "metadata.json")

        return cls(
            vectorizer=vec,
            classifier=clf,
            label_to_id=meta["label_to_id"],
            id_to_label={int(k): v for k, v in meta["id_to_label"].items()},
            config=meta["config"],
            metrics=meta.get("metrics", {}),
        )
