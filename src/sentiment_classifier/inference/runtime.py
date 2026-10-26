"""ONNX Runtime execution session manager and forward pass executor."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from sentiment_classifier.exceptions import ModelInferenceError, ModelNotFoundError


class ONNXRuntimeEngine:
    """Manages an ONNX Runtime inference session with configured threading and provider options."""

    def __init__(
        self,
        model_path: str | Path,
        execution_provider: str = "CPUExecutionProvider",
        intra_op_threads: int = 4,
        inter_op_threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelNotFoundError(f"ONNX model file not found at: {self.model_path}")

        self.execution_provider = execution_provider
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads

        # Configure SessionOptions
        self.opts = ort.SessionOptions()
        self.opts.intra_op_num_threads = intra_op_threads
        self.opts.inter_op_num_threads = inter_op_threads
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=self.opts,
            providers=[execution_provider],
        )

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run_forward(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Execute forward pass and return raw logits array [batch_size, num_classes]."""
        # Ensure only expected inputs are passed
        feed_dict = {k: inputs[k] for k in self.input_names if k in inputs}
        try:
            outputs = self.session.run(self.output_names, feed_dict)
            logits = outputs[0]
            return logits
        except Exception as e:
            raise ModelInferenceError(f"ONNX Runtime forward pass failed: {e}") from e

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
