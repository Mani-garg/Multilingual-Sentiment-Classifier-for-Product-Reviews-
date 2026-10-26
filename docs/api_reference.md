# API Reference & Serving Specification

## 1. Overview

The production serving layer is implemented as a high-performance **FastAPI** application with Pydantic v2 data models, asynchronous request handling, structured JSON logging, and dynamic micro-batching.

- **Base URL**: `http://localhost:8000`
- **Interactive Documentation (Swagger)**: `http://localhost:8000/docs`
- **Alternative Documentation (ReDoc)**: `http://localhost:8000/redoc`

---

## 2. Endpoints

### 2.1 Health Probes

#### Liveness Probe
- **Route**: `GET /health/live`
- **Description**: Returns 200 OK if the ASGI process is running and responsive.
- **Response**:
```json
{
  "status": "healthy",
  "uptime_seconds": 128.4
}
```

#### Readiness Probe
- **Route**: `GET /health/ready`
- **Description**: Returns 200 OK only after model weights, tokenizer, and ONNX Runtime sessions are fully initialized.
- **Response**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "backend": "onnx"
}
```

---

### 2.2 Model Metadata

#### Get Serving Model Metadata
- **Route**: `GET /v1/models/metadata`
- **Description**: Inspect model name, active runtime backend, supported languages, label mappings, and max sequence length.
- **Response**:
```json
{
  "model_name": "multilingual-sentiment-classifier",
  "version": "1.0.0",
  "backend": "onnx",
  "supported_languages": ["en", "de", "es", "fr", "ja", "zh"],
  "labels": ["negative", "neutral", "positive"],
  "max_length": 256
}
```

---

### 2.3 Single Sentiment Prediction

- **Route**: `POST /v1/predict`
- **Content-Type**: `application/json`

#### Request Body
```json
{
  "text": "The battery life is phenomenal and build quality is top-notch!",
  "language": "en"
}
```
*Note: `language` is optional; if omitted, the system will detect it automatically.*

#### Response Body
```json
{
  "label": "positive",
  "confidence": 0.9842,
  "scores": {
    "negative": 0.0051,
    "neutral": 0.0107,
    "positive": 0.9842
  },
  "detected_language": "en",
  "latency_ms": 3.12,
  "model_version": "1.0.0",
  "warnings": []
}
```

---

### 2.4 Batch Sentiment Prediction

- **Route**: `POST /v1/predict/batch`
- **Content-Type**: `application/json`

#### Request Body
```json
{
  "texts": [
    "Das Produkt funktioniert super und kam schnell an.",
    "Ce n'est pas bon du tout, je suis très déçu.",
    "画質が綺麗で大満足です！"
  ]
}
```

#### Response Body
```json
{
  "predictions": [
    {
      "label": "positive",
      "confidence": 0.9610,
      "scores": {"negative": 0.012, "neutral": 0.027, "positive": 0.961},
      "detected_language": "de",
      "latency_ms": 4.15,
      "model_version": "1.0.0",
      "warnings": []
    },
    {
      "label": "negative",
      "confidence": 0.9450,
      "scores": {"negative": 0.945, "neutral": 0.041, "positive": 0.014},
      "detected_language": "fr",
      "latency_ms": 4.15,
      "model_version": "1.0.0",
      "warnings": []
    },
    {
      "label": "positive",
      "confidence": 0.9820,
      "scores": {"negative": 0.006, "neutral": 0.012, "positive": 0.982},
      "detected_language": "ja",
      "latency_ms": 4.15,
      "model_version": "1.0.0",
      "warnings": []
    }
  ],
  "total_samples": 3,
  "total_latency_ms": 4.5
}
```

---

## 3. Error Responses

The API uses standard HTTP status codes:
- `400 Bad Request`: Malformed payload or validation error.
- `422 Unprocessable Content`: Empty text, whitespace-only text, text exceeding 10,000 characters, or batch size exceeding 128 items.
- `503 Service Unavailable`: Model is still warming up and not ready for traffic.
