"""Safe serialization and deserialization utilities for configs, metrics, and models."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentiment_classifier.exceptions import ModelNotFoundError


class JSONEncoderWithNumpy(json.JSONEncoder):
    """JSON Encoder that converts numpy scalars and arrays to native Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: dict[str, Any] | list[Any], path: str | Path, indent: int = 2) -> None:
    """Save dictionary or list to JSON file with numpy support."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=JSONEncoderWithNumpy, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load and parse JSON file."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {target_path}")
    with open(target_path, encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, path: str | Path) -> None:
    """Serialize scikit-learn object using joblib."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, target_path)


def load_joblib(path: str | Path) -> Any:
    """Deserialize scikit-learn object using joblib."""
    target_path = Path(path)
    if not target_path.exists():
        raise ModelNotFoundError(f"Joblib artifact not found: {target_path}")
    return joblib.load(target_path)
