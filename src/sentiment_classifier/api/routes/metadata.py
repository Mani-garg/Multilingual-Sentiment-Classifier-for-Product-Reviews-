"""Model metadata and discovery routes."""

from fastapi import APIRouter, Depends

from sentiment_classifier.api.dependencies import get_app_settings, get_predictor
from sentiment_classifier.config import AppSettings
from sentiment_classifier.inference.predictor import SentimentPredictor
from sentiment_classifier.inference.schemas import ModelInfoResponse
from sentiment_classifier.labels import SUPPORTED_LABELS, SUPPORTED_LANGUAGES

router = APIRouter(prefix="/api/v1/model", tags=["Model Information"])


@router.get("/info", response_model=ModelInfoResponse, summary="Model Metadata & Capabilities")
async def get_model_info(
    settings: AppSettings = Depends(get_app_settings),
    predictor: SentimentPredictor = Depends(get_predictor),
) -> ModelInfoResponse:
    """Retrieve non-sensitive information regarding active model version, language support, and backend."""
    return ModelInfoResponse(
        model_name="Multilingual Sentiment Classifier for Product Reviews",
        model_version=predictor.model_version,
        inference_backend=predictor.backend,
        checkpoint_base=settings.model_checkpoint,
        supported_labels=SUPPORTED_LABELS,
        supported_languages=sorted(SUPPORTED_LANGUAGES),
        max_sequence_length=settings.max_sequence_length,
        execution_provider=settings.onnx_execution_provider,
        quantized=(predictor.backend == "quantized_onnx"),
    )
