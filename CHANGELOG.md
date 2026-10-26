# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-26

### Added
- Complete multilingual sentiment classification architecture supporting English, German, Spanish, French, Japanese, and Chinese.
- Data ingestion, provenance verification, conservative cleaning, deduplication, and leakage-resistant stratified splitting.
- TF-IDF + Logistic Regression linear baseline with full per-class and per-language metrics.
- Multilingual transformer fine-tuning pipeline with dynamic padding, class weighting, and model checkpointing.
- Targeted data augmentation module focusing on negation preservation and contextual paraphrase on training split only.
- Diagnostic error-analysis engine categorizing negation failures, sarcasm, mixed sentiment, and language-specific nuances.
- Tested ONNX graph export and dynamic int8 quantization pipeline.
- Numerical parity validation comparing PyTorch/TensorFlow reference logits with ONNX Runtime predictions.
- Production-grade FastAPI serving layer with micro-batching, typed Pydantic contracts, health checks, and OpenAPI docs.
- Latency and throughput benchmarking suite measuring cold-start, p50, p95, p99, memory footprint, and disk size.
- Comprehensive automated test suite spanning unit tests, integration pipelines, API security, and optimization parity.
- Containerized Docker deployment and multi-service Docker Compose configuration.
