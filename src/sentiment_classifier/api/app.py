"""FastAPI Application factory and lifecycle configuration."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from sentiment_classifier.api.errors import register_error_handlers
from sentiment_classifier.api.routes import health, metadata, predict
from sentiment_classifier.config import get_settings
from sentiment_classifier.inference.batching import DynamicBatcher
from sentiment_classifier.inference.predictor import SentimentPredictor
from sentiment_classifier.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: Initialize predictor and dynamic batcher; clean shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level, use_json=(settings.app_env == "production"))
    logger.info(
        "Starting Multilingual Sentiment Classifier API (Backend: %s)...",
        settings.inference_backend,
    )

    # Initialize predictor
    predictor = SentimentPredictor(settings=settings)
    app.state.predictor = predictor

    # Initialize dynamic batcher if batching wait interval > 0
    if settings.dynamic_batch_max_wait_ms > 0 and predictor.is_ready:
        app.state.batcher = DynamicBatcher(
            predict_fn=predictor.predict_batch,
            max_batch_size=settings.inference_batch_size,
            max_wait_ms=settings.dynamic_batch_max_wait_ms,
        )
    else:
        app.state.batcher = None

    yield

    # Clean shutdown
    if getattr(app.state, "batcher", None) is not None:
        app.state.batcher.shutdown()
    logger.info("API shutdown complete.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="Multilingual Sentiment Classifier for Product Reviews",
        description=(
            "High-throughput, production-ready sentiment classification REST API "
            "supporting multiple languages, dynamic micro-batching, and optimized ONNX Runtime."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Correlation ID & Latency logging middleware
    @app.middleware("http")
    async def request_correlation_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"

        # Structured diagnostic log (never logging body)
        logger.info(
            "%s %s -> %d (%.2f ms) [ReqId: %s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response

    # Register error handlers
    register_error_handlers(app)

    # Include routers
    app.include_router(health.router)
    app.include_router(metadata.router)
    app.include_router(predict.router)

    return app


app = create_app()
