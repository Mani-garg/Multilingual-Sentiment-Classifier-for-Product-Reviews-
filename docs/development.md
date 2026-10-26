# Developer Guide & Contributing Workflow

## 1. Local Environment Setup

Ensure you have Python 3.11 installed. We recommend `uv` for package management:

```bash
# Clone the repository
git clone https://github.com/Mani-garg/Multilingual-Sentiment-Classifier-for-Product-Reviews-.git
cd Multilingual-Sentiment-Classifier-for-Product-Reviews-

# Create virtual environment with Python 3.11
uv venv .venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
uv pip install -e ".[dev]"
```

---

## 2. Running Quality Checks

### Linting and Formatting
```bash
# Check code style with Ruff
ruff check src tests scripts

# Auto-fix linting issues
ruff check --fix src tests scripts

# Format code
ruff format src tests scripts
```

### Running Test Suite
```bash
# Run all unit, integration, API, security, and performance tests
pytest tests -v

# Run only unit tests
pytest tests/unit -v

# Run with test coverage
pytest --cov=sentiment_classifier --cov-report=term-missing tests/
```

---

## 3. End-to-End Pipeline Execution

Execute the full demonstration script to verify data generation, baseline training, augmentation ablation, ONNX export, and inference benchmarks:

```bash
python scripts/demo.py
```

---

## 4. Serving Locally

Start the FastAPI application with auto-reload:

```bash
uvicorn sentiment_classifier.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Test single prediction:
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Extremely fast delivery and crystal clear display!", "language": "en"}'
```
