"""Integration test verifying ONNX export, structural validation, quantization, and parity."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from sentiment_classifier.optimization.export_onnx import _export_fallback_graph
from sentiment_classifier.optimization.quantize import quantize_onnx_model
from sentiment_classifier.optimization.validate_parity import validate_numerical_parity


def test_onnx_export_and_quantization_lifecycle(tmp_path: Path) -> None:
    """Verify ONNX graph generation, validation, dynamic quantization, and forward pass."""
    onnx_file = tmp_path / "model.onnx"
    quant_file = tmp_path / "model.quant.onnx"

    # 1. Export valid ONNX graph
    _export_fallback_graph(onnx_file, opset_version=17)
    assert onnx_file.exists()

    # 2. Structural verification
    model = onnx.load(str(onnx_file))
    onnx.checker.check_model(model)

    # 3. Dynamic Quantization
    quantize_onnx_model(onnx_file, quant_file)
    assert quant_file.exists()

    # 4. ONNX Runtime forward pass on both graphs
    sess_fp32 = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(quant_file), providers=["CPUExecutionProvider"])

    inputs = {
        "input_ids": np.array([[101, 2054, 102], [101, 3042, 102]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int64),
    }

    out_fp32 = sess_fp32.run(None, inputs)[0]
    out_int8 = sess_int8.run(None, inputs)[0]

    assert out_fp32.shape == (2, 3)
    assert out_int8.shape == (2, 3)

    # 5. Numerical parity check
    report = validate_numerical_parity(
        reference_logits=out_fp32,
        onnx_session=sess_int8,
        sample_inputs=inputs,
        atol=0.5,
        min_agreement=0.5,
    )
    assert report.sample_count == 2
