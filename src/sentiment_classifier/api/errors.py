"""Standardized error handlers and sanitization for FastAPI."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sentiment_classifier.exceptions import ModelNotFoundError, SentimentClassifierError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Register application-wide exception handlers with safe error serialization."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Client request validation error on %s: %s", request.url.path, exc.errors())
        # Strip out raw values that could leak private review contents
        cleaned_errors = []
        for err in exc.errors():
            cleaned_errors.append(
                {
                    "loc": err.get("loc"),
                    "msg": err.get("msg"),
                    "type": err.get("type"),
                }
            )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "message": "Input failed schema validation rules.",
                "details": cleaned_errors,
            },
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        logger.error("Model unavailable on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Service Unavailable",
                "message": exc.message,
            },
        )

    @app.exception_handler(SentimentClassifierError)
    async def domain_error_handler(request: Request, exc: SentimentClassifierError) -> JSONResponse:
        logger.error("Domain exception on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception on %s", request.url.path)
        # Avoid exposing internal tracebacks or stack traces to client
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while processing the request.",
            },
        )
