"""Unified predictor interface abstracting baseline, ONNX, and quantized serving backends."""

import numpy as np

from sentiment_classifier.baseline.artifacts import BaselineArtifacts
from sentiment_classifier.baseline.predict import BaselinePredictor
from sentiment_classifier.config import AppSettings, get_settings
from sentiment_classifier.exceptions import ModelInferenceError, ModelNotFoundError
from sentiment_classifier.inference.runtime import ONNXRuntimeEngine
from sentiment_classifier.inference.schemas import SentimentResponse
from sentiment_classifier.inference.tokenizer_service import TokenizerService
from sentiment_classifier.labels import ID_TO_LABEL, SUPPORTED_LABELS
from sentiment_classifier.preprocessing.language_metadata import detect_or_verify_language


class SentimentPredictor:
    """Unified predictor supporting ONNX, Quantized ONNX, and Baseline backends."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.inference_backend
        self.model_version = "1.0.0"

        self.tokenizer_service: TokenizerService | None = None
        self.onnx_engine: ONNXRuntimeEngine | None = None
        self.baseline_predictor: BaselinePredictor | None = None
        self.is_ready = False
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the configured backend engine."""
        if self.backend in ["onnx", "quantized_onnx"]:
            onnx_file = (
                self.settings.onnx_artifact_dir / "model.quant.onnx"
                if self.backend == "quantized_onnx"
                else self.settings.onnx_artifact_dir / "model.onnx"
            )

            # If specific quantized file doesn't exist, fallback to base model.onnx
            if not onnx_file.exists() and (self.settings.onnx_artifact_dir / "model.onnx").exists():
                onnx_file = self.settings.onnx_artifact_dir / "model.onnx"

            if onnx_file.exists():
                self.onnx_engine = ONNXRuntimeEngine(
                    model_path=onnx_file,
                    execution_provider=self.settings.onnx_execution_provider,
                    intra_op_threads=self.settings.num_inference_threads,
                )
                self.tokenizer_service = TokenizerService(
                    checkpoint=self.settings.model_checkpoint,
                    max_length=self.settings.max_sequence_length,
                )
                self.is_ready = True
            else:
                self.is_ready = False

        elif self.backend == "baseline":
            baseline_dir = self.settings.baseline_artifact_dir
            if (baseline_dir / "classifier.joblib").exists():
                artifacts = BaselineArtifacts.load(baseline_dir)
                self.baseline_predictor = BaselinePredictor(artifacts)
                self.is_ready = True
            else:
                self.is_ready = False

    def predict_batch(
        self,
        texts: list[str],
        languages: list[str | None] | None = None,
    ) -> list[SentimentResponse]:
        """Perform batched prediction across inputs returning strongly typed SentimentResponses."""
        if not self.is_ready:
            raise ModelNotFoundError(
                f"Predictor is not ready. Backend '{self.backend}' artifacts missing. "
                f"Please ensure model is trained and exported."
            )

        if not texts:
            return []

        if languages is None:
            languages = [None] * len(texts)

        # Baseline execution path
        if self.backend == "baseline" and self.baseline_predictor is not None:
            baseline_results = self.baseline_predictor.predict(texts)
            responses = []
            for res, lang in zip(baseline_results, languages, strict=False):
                lang_meta = detect_or_verify_language(res["label"], supplied_lang=lang)
                warnings = []
                if not lang_meta.is_supported:
                    warnings.append(f"Language '{lang}' is not in verified model coverage.")

                responses.append(
                    SentimentResponse(
                        label=res["label"],
                        scores=res["scores"],
                        model_version="baseline-1.0.0",
                        inference_backend="baseline",
                        warnings=warnings,
                    )
                )
            return responses

        # ONNX / Quantized ONNX execution path
        if self.onnx_engine is None or self.tokenizer_service is None:
            raise ModelInferenceError("ONNX Runtime engine not loaded.")

        inputs, per_sample_warnings = self.tokenizer_service.tokenize_batch(texts)
        logits = self.onnx_engine.run_forward(inputs)
        probs = self.onnx_engine.softmax(logits)
        preds = np.argmax(probs, axis=-1)

        responses = []
        for i, (pred_idx, prob_row) in enumerate(zip(preds, probs, strict=False)):
            label = ID_TO_LABEL.get(int(pred_idx), "neutral")
            scores = {ID_TO_LABEL[j]: float(prob_row[j]) for j in range(len(SUPPORTED_LABELS))}

            warnings = list(per_sample_warnings[i])
            supplied_l = languages[i]
            if supplied_l:
                meta = detect_or_verify_language(texts[i], supplied_lang=supplied_l)
                if not meta.is_supported:
                    warnings.append(f"Language '{supplied_l}' is outside verified evaluation set.")

            responses.append(
                SentimentResponse(
                    label=label,
                    scores=scores,
                    model_version=self.model_version,
                    inference_backend=self.backend,
                    warnings=warnings,
                )
            )

        return responses

    def predict_single(self, text: str, language: str | None = None) -> SentimentResponse:
        """Classify a single review text."""
        results = self.predict_batch([text], [language])
        return results[0]
