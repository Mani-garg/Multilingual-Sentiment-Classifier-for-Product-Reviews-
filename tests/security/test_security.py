"""Security tests verifying rejection of malformed, oversized, or unauthorized inputs."""

from fastapi.testclient import TestClient

from sentiment_classifier.api.app import app

client = TestClient(app)


def test_reject_oversized_text() -> None:
    """Inputs exceeding 5,000 characters must be rejected with HTTP 422."""
    giant_text = "excellent " * 700  # > 5000 chars
    res = client.post("/api/v1/sentiment/predict", json={"text": giant_text})
    assert res.status_code == 422
    data = res.json()
    assert "Validation Error" in data["error"]


def test_reject_whitespace_only_text() -> None:
    """Whitespace-only text must be rejected with HTTP 422."""
    res = client.post("/api/v1/sentiment/predict", json={"text": "   \n\t  "})
    assert res.status_code == 422
    data = res.json()
    assert "Validation Error" in data["error"]


def test_reject_empty_text() -> None:
    """Empty string must be rejected with HTTP 422."""
    res = client.post("/api/v1/sentiment/predict", json={"text": ""})
    assert res.status_code == 422


def test_reject_oversized_batch() -> None:
    """Batches exceeding 64 items must be rejected with HTTP 422."""
    items = [{"text": f"Review {i}"} for i in range(70)]
    res = client.post("/api/v1/sentiment/predict/batch", json={"reviews": items})
    assert res.status_code == 422


def test_error_responses_do_not_leak_stack_traces() -> None:
    """500 error responses must not leak tracebacks or file paths to client."""
    from sentiment_classifier.exceptions import ModelNotFoundError

    class UnreadyPredictor:
        is_ready = False
        backend = "onnx"

        def predict_single(self, text: str, language: str | None = None):
            raise ModelNotFoundError("Model artifact missing.")

    app.state.predictor = UnreadyPredictor()
    res = client.post("/api/v1/sentiment/predict", json={"text": "Valid review text"})
    assert res.status_code in [500, 503]
    data = res.json()
    # Confirm no stack trace / traceback keyword in response
    assert "Traceback" not in str(data)
    assert "File " not in str(data)
