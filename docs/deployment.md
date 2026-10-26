# Deployment & Operations Guide

## 1. Containerization Architecture

The system is containerized with a lean, security-hardened `Dockerfile` using `python:3.11-slim`:
- Non-root user `appuser` (UID 1000) execution to mitigate container escape vulnerabilities.
- Multi-stage build layer caching to ensure fast rebuilds.
- `uv` package manager for deterministic dependency resolution.
- Native ONNX Runtime CPU acceleration.

---

## 2. Docker Deployment

### 2.1 Build Container Image
```bash
docker build -t sentiment-classifier:latest .
```

### 2.2 Run Container
```bash
docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  -e LOG_LEVEL=INFO \
  -e MAX_BATCH_SIZE=32 \
  -e BACKEND=onnx \
  sentiment-classifier:latest
```

### 2.3 Verify Health
```bash
curl http://localhost:8000/health/ready
```

---

## 3. Docker Compose Deployment

A multi-container setup is orchestrated via `compose.yaml`:

```bash
# Start service in detached mode
docker compose up -d

# Stream application logs
docker compose logs -f api

# Stop and remove containers
docker compose down
```

---

## 4. Continuous Integration (GitHub Actions)

The CI workflow in `.github/workflows/ci.yml` triggers on every push and pull request to `main`:
1. **Lint & Code Style**: Runs `ruff check` and `ruff format --check`.
2. **Type Checking**: Validates type signatures.
3. **Unit & Integration Tests**: Executes `pytest tests -v` with coverage reporting.
4. **Parity & Benchmark Sanity**: Verifies model export and latency bounds.
