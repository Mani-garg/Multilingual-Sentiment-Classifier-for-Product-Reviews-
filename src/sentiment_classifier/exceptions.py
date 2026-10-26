"""Custom exception hierarchy for the sentiment classification package."""


class SentimentClassifierError(Exception):
    """Base exception for all domain errors within the sentiment classifier."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(SentimentClassifierError):
    """Raised when dataset schema, column requirements, or content invariants fail."""

    pass


class ModelNotFoundError(SentimentClassifierError):
    """Raised when model weights or ONNX graph cannot be found at configured path."""

    pass


class ModelInferenceError(SentimentClassifierError):
    """Raised when runtime inference forward pass or tensor conversion fails."""

    pass


class ConfigurationError(SentimentClassifierError):
    """Raised when invalid or conflicting settings are provided."""

    pass


class PreprocessingError(SentimentClassifierError):
    """Raised when text normalization or tokenization fails unexpectedly."""

    pass


class ExportError(SentimentClassifierError):
    """Raised when ONNX conversion or quantization fails."""

    pass


class ServiceUnavailableError(SentimentClassifierError):
    """Raised when the serving runtime is not initialized or model is not loaded."""

    pass
