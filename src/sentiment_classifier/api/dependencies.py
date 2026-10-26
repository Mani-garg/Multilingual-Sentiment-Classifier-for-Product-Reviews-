"""Dependency injection providers for FastAPI routes."""

from fastapi import Request

from sentiment_classifier.config import AppSettings, get_settings
from sentiment_classifier.inference.predictor import SentimentPredictor


def get_app_settings() -> AppSettings:
    """Return application configuration settings."""
    return get_settings()


def get_predictor(request: Request) -> SentimentPredictor:
    """Extract model predictor instance from application state."""
    return request.app.state.predictor
