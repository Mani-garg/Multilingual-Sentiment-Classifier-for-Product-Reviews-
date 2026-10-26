"""Inference prediction routes for single reviews and bounded micro-batches."""

import time

from fastapi import APIRouter, Depends, Request

from sentiment_classifier.api.dependencies import get_predictor
from sentiment_classifier.inference.predictor import SentimentPredictor
from sentiment_classifier.inference.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    SentimentRequest,
    SentimentResponse,
)

router = APIRouter(prefix="/api/v1/sentiment", tags=["Sentiment Inference"])


@router.post(
    "/predict",
    response_model=SentimentResponse,
    summary="Classify Single Review Sentiment",
)
async def predict_single_sentiment(
    payload: SentimentRequest,
    request: Request,
    predictor: SentimentPredictor = Depends(get_predictor),
) -> SentimentResponse:
    """Classify the sentiment expressed in a single product review.

    Optionally accepts an ISO language code for validation and auditing.
    """
    # If dynamic micro-batcher is configured in app state, route through it
    if hasattr(request.app.state, "batcher") and request.app.state.batcher is not None:
        return await request.app.state.batcher.submit(payload.text, payload.language)

    return predictor.predict_single(payload.text, payload.language)


@router.post(
    "/predict/batch",
    response_model=BatchSentimentResponse,
    summary="Classify Batch of Reviews",
)
async def predict_batch_sentiment(
    payload: BatchSentimentRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
) -> BatchSentimentResponse:
    """Classify sentiment across a bounded batch of reviews (1 to 64 items).

    Guarantees that responses correspond strictly to the submitted order.
    """
    start_time = time.perf_counter()
    texts = [r.text for r in payload.reviews]
    languages = [r.language for r in payload.reviews]

    predictions = predictor.predict_batch(texts, languages)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return BatchSentimentResponse(
        predictions=predictions,
        batch_size=len(predictions),
        total_latency_ms=round(elapsed_ms, 2),
    )
