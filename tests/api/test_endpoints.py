"""API endpoints integration tests using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from sentiment_classifier.api.app import app


@pytest.fixture
def client(tmp_path) -> TestClient:
    """TestClient fixture with mock predictor attached."""
    from sentiment_classifier.inference.schemas import SentimentResponse

    test_client = TestClient(app)

    # Attach dummy predictor to app.state
    class MockPredictor:
        is_ready = True
        backend = "mock_onnx"
        model_version = "1.0.0-test"

        def predict_single(self, text: str, language: str | None = None) -> SentimentResponse:
            return SentimentResponse(
                label="positive"
                if "good" in text.lower() or "great" in text.lower()
                else "negative",
                scores={"positive": 0.85, "neutral": 0.10, "negative": 0.05},
                model_version=self.model_version,
                inference_backend=self.backend,
                warnings=[],
            )

        def predict_batch(
            self, texts: list[str], languages: list[str | None] | None = None
        ) -> list[SentimentResponse]:
            return [
                self.predict_single(t, lang)
                for t, lang in zip(texts, languages or [None] * len(texts), strict=False)
            ]

    app.state.predictor = MockPredictor()
    return test_client


def test_health_live(client: TestClient) -> None:
    """Liveness probe must return 200 and 'ok' status."""
    res = client.get("/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_health_ready(client: TestClient) -> None:
    """Readiness probe must return 200 when model is loaded."""
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["ready"] is True


def test_model_info(client: TestClient) -> None:
    """Metadata route must expose supported labels and runtime backend."""
    res = client.get("/api/v1/model/info")
    assert res.status_code == 200
    data = res.json()
    assert "positive" in data["supported_labels"]
    assert "en" in data["supported_languages"]
    assert data["model_version"] == "1.0.0-test"


def test_predict_single_endpoint(client: TestClient) -> None:
    """Single prediction endpoint must return structured label and probability scores."""
    payload = {
        "text": "This product is great and fast!",
        "language": "en",
    }
    res = client.post("/api/v1/sentiment/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "positive"
    assert "scores" in data
    assert pytest.approx(sum(data["scores"].values()), abs=1e-2) == 1.0


def test_predict_batch_endpoint_order_preservation(client: TestClient) -> None:
    """Batch endpoint must return predictions in strictly preserved order."""
    payload = {
        "reviews": [
            {"text": "Great device", "language": "en"},
            {"text": "Terrible failure", "language": "en"},
            {"text": "Another good phone", "language": "en"},
        ]
    }
    res = client.post("/api/v1/sentiment/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["batch_size"] == 3
    preds = data["predictions"]
    assert preds[0]["label"] == "positive"
    assert preds[1]["label"] == "negative"
    assert preds[2]["label"] == "positive"
