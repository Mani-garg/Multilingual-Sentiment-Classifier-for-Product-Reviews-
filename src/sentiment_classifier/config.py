"""Centralized application and experiment configuration using Pydantic Settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Strongly-typed application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core
    app_env: str = Field(default="development", description="Runtime environment name")
    log_level: str = Field(default="INFO", description="Logging severity level")
    random_seed: int = Field(default=42, description="Global random seed for reproducibility")

    # Dataset Paths and Provenance
    dataset_source_id: str = Field(default="amazon_reviews_multi")
    dataset_revision: str = Field(default="1.0.0")
    raw_data_path: Path = Field(default=Path("data/raw/multilingual_reviews.jsonl"))
    interim_data_path: Path = Field(default=Path("data/interim/cleaned_reviews.jsonl"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    sample_data_path: Path = Field(default=Path("data/sample/multilingual_reviews_sample.jsonl"))

    # Model & Tokenizer
    model_checkpoint: str = Field(default="distilbert-base-multilingual-cased")
    model_revision: str = Field(default="main")
    max_sequence_length: int = Field(default=128, ge=16, le=512)

    # Training Hyperparameters
    train_batch_size: int = Field(default=32, ge=1)
    eval_batch_size: int = Field(default=64, ge=1)
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=3e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    mixed_precision: bool = Field(default=False)
    early_stopping_patience: int = Field(default=2, ge=1)
    use_class_weights: bool = Field(default=True)

    # Artifact Storage Directories
    baseline_artifact_dir: Path = Field(default=Path("artifacts/baseline"))
    transformer_artifact_dir: Path = Field(default=Path("artifacts/transformer"))
    onnx_artifact_dir: Path = Field(default=Path("artifacts/onnx"))
    reports_dir: Path = Field(default=Path("artifacts/reports"))

    # ONNX & Runtime Optimization
    onnx_execution_provider: str = Field(default="CPUExecutionProvider")
    onnx_opset_version: int = Field(default=17, ge=11)
    quantization_mode: str = Field(default="dynamic_int8")
    inference_backend: str = Field(
        default="onnx"
    )  # onnx | quantized_onnx | transformer_reference | baseline
    num_inference_threads: int = Field(default=4, ge=1)

    # Serving & API Limits
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    max_request_text_length: int = Field(default=5000, ge=10)
    max_texts_per_request: int = Field(default=64, ge=1)
    max_concurrent_requests: int = Field(default=10, ge=1)
    inference_batch_size: int = Field(default=32, ge=1)
    dynamic_batch_max_wait_ms: int = Field(default=20, ge=0)
    request_timeout_seconds: float = Field(default=10.0, gt=0.0)
    api_key_enabled: bool = Field(default=False)
    api_key: str | None = Field(default=None)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("inference_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        valid_backends = {"onnx", "quantized_onnx", "transformer_reference", "baseline"}
        norm = v.lower()
        if norm not in valid_backends:
            raise ValueError(f"Invalid inference_backend '{v}'. Must be one of {valid_backends}")
        return norm


def get_settings() -> AppSettings:
    """Instantiate and return settings singleton."""
    return AppSettings()
