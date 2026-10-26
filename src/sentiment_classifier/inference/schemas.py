"""Pydantic request and response schemas for sentiment inference and health endpoints."""

from pydantic import BaseModel, Field, field_validator


class SentimentRequest(BaseModel):
    """Payload for single-review sentiment classification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw review text content to classify.",
        examples=["The product exceeded all my expectations, highly recommended!"],
    )
    language: str | None = Field(
        default=None,
        max_length=10,
        description="Optional ISO 639-1 language code (e.g., 'en', 'de', 'es', 'fr', 'ja', 'zh').",
        examples=["en"],
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Review text cannot be empty or contain only whitespace.")
        return stripped

    @field_validator("language")
    @classmethod
    def normalize_language_code(cls, v: str | None) -> str | None:
        if v is not None:
            norm = v.strip().lower()
            if not norm:
                return None
            return norm
        return None


class BatchSentimentRequest(BaseModel):
    """Payload for bounded batch review sentiment classification."""

    reviews: list[SentimentRequest] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Bounded list of review items for batched classification.",
    )


class SentimentResponse(BaseModel):
    """Prediction outcome including class probabilities, version, and warnings."""

    label: str = Field(..., description="Predicted sentiment class (negative, neutral, positive).")
    scores: dict[str, float] = Field(
        ...,
        description="Normalized posterior class probabilities mapping label to confidence in [0, 1].",
    )
    model_version: str = Field(..., description="Semantic version or hash of active model.")
    inference_backend: str = Field(
        ..., description="Serving engine ('onnx', 'quantized_onnx', 'baseline')."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings (e.g. text truncation)."
    )


class BatchSentimentResponse(BaseModel):
    """Ordered batch prediction results."""

    predictions: list[SentimentResponse] = Field(
        ..., description="List of prediction outputs in exact request order."
    )
    batch_size: int = Field(..., description="Total reviews processed in batch.")
    total_latency_ms: float = Field(
        ..., description="Server-side inference execution duration in milliseconds."
    )


class HealthResponse(BaseModel):
    """Liveness check status."""

    status: str = Field(default="ok", description="Liveness indicator.")
    environment: str = Field(..., description="Running environment.")


class ReadinessResponse(BaseModel):
    """Readiness probe indicating model readiness and memory health."""

    ready: bool = Field(..., description="Whether model inference backend is loaded and ready.")
    backend: str = Field(..., description="Configured inference backend.")
    model_loaded: bool = Field(..., description="Status of artifact loading.")
    error: str | None = Field(default=None, description="Details if service is not ready.")


class ModelInfoResponse(BaseModel):
    """Detailed model metadata, supported labels, and runtime environment."""

    model_name: str
    model_version: str
    inference_backend: str
    checkpoint_base: str
    supported_labels: list[str]
    supported_languages: list[str]
    max_sequence_length: int
    execution_provider: str
    quantized: bool
