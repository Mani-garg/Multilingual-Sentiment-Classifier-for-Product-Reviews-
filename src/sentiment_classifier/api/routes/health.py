"""Health, liveness, and readiness probe endpoints."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from sentiment_classifier.api.dependencies import get_app_settings, get_predictor
from sentiment_classifier.config import AppSettings
from sentiment_classifier.inference.predictor import SentimentPredictor
from sentiment_classifier.inference.schemas import HealthResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", response_model=HealthResponse, summary="Liveness Probe")
async def liveness_check(settings: AppSettings = Depends(get_app_settings)) -> HealthResponse:
    """Simple check to confirm the service process is up and responsive."""
    return HealthResponse(status="ok", environment=settings.app_env)


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness Probe")
async def readiness_check(
    predictor: SentimentPredictor = Depends(get_predictor),
) -> ReadinessResponse:
    """Verify that required model weights and inference runtime dependencies are loaded and ready."""
    is_ready = predictor.is_ready
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content=ReadinessResponse(
            ready=is_ready,
            backend=predictor.backend,
            model_loaded=is_ready,
            error=None
            if is_ready
            else f"Backend '{predictor.backend}' artifact not loaded or missing on disk.",
        ).model_dump(),
    )
