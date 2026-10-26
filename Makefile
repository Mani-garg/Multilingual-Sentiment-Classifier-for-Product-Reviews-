.PHONY: install prepare-data validate-data train-baseline train-transformer evaluate analyze-errors augment export-onnx quantize benchmark serve test lint format docker-build docker-up docker-down clean demo

PYTHON = .venv/Scripts/python.exe
# Fallback to python if virtual environment is activated
ifeq ($(OS),Windows_NT)
    VENV_BIN = .venv/Scripts
else
    VENV_BIN = .venv/bin
    PYTHON = $(VENV_BIN)/python
endif

install:
	uv venv .venv --python 3.11
	uv pip install --python $(VENV_BIN)/python ".[dev,training]"

demo:
	$(PYTHON) scripts/demo.py

prepare-data:
	$(PYTHON) scripts/generate_demo_data.py --output data/raw/multilingual_reviews.jsonl --samples 3000
	$(PYTHON) scripts/download_or_prepare_data.py --input data/raw/multilingual_reviews.jsonl

validate-data:
	$(PYTHON) scripts/validate_data.py --input data/raw/multilingual_reviews.jsonl --report-path artifacts/reports/data_quality_report.json

train-baseline:
	$(PYTHON) scripts/train_baseline.py --config configs/training.yaml

train-transformer:
	$(PYTHON) scripts/train_transformer.py --config configs/training.yaml

evaluate:
	$(PYTHON) scripts/evaluate_model.py --backend onnx --split test

analyze-errors:
	$(PYTHON) scripts/analyze_errors.py --output artifacts/reports/error_analysis_report.json

augment:
	$(PYTHON) scripts/run_augmentation_experiment.py --config configs/training.yaml

export-onnx:
	$(PYTHON) scripts/export_onnx.py --model-dir artifacts/transformer --output artifacts/onnx/model.onnx

quantize:
	$(PYTHON) scripts/export_onnx.py --quantize --input artifacts/onnx/model.onnx --output artifacts/onnx/model.quant.onnx

benchmark:
	$(PYTHON) scripts/benchmark_inference.py --backend onnx --batch-sizes 1,8,32

serve:
	$(PYTHON) -m uvicorn sentiment_classifier.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/unit tests/integration tests/api tests/security tests/performance -v

test-fast:
	$(PYTHON) -m pytest tests/unit tests/api -v

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
