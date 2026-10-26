"""FastAPI serving package."""

from sentiment_classifier.api.app import app, create_app

__all__ = ["app", "create_app"]
