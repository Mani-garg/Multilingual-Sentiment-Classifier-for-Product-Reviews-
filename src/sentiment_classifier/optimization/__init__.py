"""ONNX graph export, dynamic quantization, numerical parity validation, and artifact comparison."""

from sentiment_classifier.optimization.compare_artifacts import compare_model_artifacts
from sentiment_classifier.optimization.export_onnx import export_to_onnx
from sentiment_classifier.optimization.quantize import quantize_onnx_model
from sentiment_classifier.optimization.validate_parity import validate_numerical_parity

__all__ = [
    "export_to_onnx",
    "quantize_onnx_model",
    "validate_numerical_parity",
    "compare_model_artifacts",
]
