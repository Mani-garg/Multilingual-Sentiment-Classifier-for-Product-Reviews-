# Contributing Guidelines

Thank you for your interest in contributing to the **Multilingual Sentiment Classifier for Product Reviews** project.

## Code Standards and Engineering Principles

1. **Reproducibility First**:
   - Random seeds must be documented and deterministic options preserved where feasible.
   - Experiments must output structured JSON metadata records including split fingerprints and hyperparameter hashes.

2. **No Data Leakage**:
   - Augmentations, vocabulary extraction, normalization statistics, and vectorizer fitting must strictly be performed on the training partition only.
   - Never use validation or test splits for hyperparameter selection, prompt tuning, or augmentation.

3. **Honest Benchmarking**:
   - Never fabricate results, benchmark metrics, or execution times.
   - All documented figures in README or reports must correspond to actual runs with recorded timestamps and hardware details.

4. **Code Quality**:
   - Run `ruff check .` and `ruff format --check .` before submitting PRs.
   - All public functions and classes must include clear type hints and docstrings.
   - Maintain unit tests in `tests/unit/` and integration tests in `tests/integration/`.

## Local Development Setup

1. Clone repository:
   ```bash
   git clone <repo-url>
   cd multilingual_sentiment_classifier
   ```

2. Create and activate virtual environment:
   ```bash
   uv venv .venv --python 3.11
   .venv\Scripts\activate  # Windows
   # or source .venv/bin/activate  # Linux/macOS
   ```

3. Install dependencies:
   ```bash
   uv pip install -e ".[dev,training]"
   ```

4. Run tests:
   ```bash
   pytest
   ```
