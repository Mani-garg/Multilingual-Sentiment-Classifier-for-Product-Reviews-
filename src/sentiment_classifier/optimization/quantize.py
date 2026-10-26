"""Dynamic INT8 quantization module for ONNX Runtime models."""

import logging
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from sentiment_classifier.exceptions import ExportError
from sentiment_classifier.utils.serialization import save_json

logger = logging.getLogger(__name__)


def quantize_onnx_model(
    input_model_path: str | Path,
    output_model_path: str | Path,
    per_channel: bool = False,
    weight_type: str = "qint8",
) -> Path:
    """Apply dynamic INT8 quantization to an exported ONNX model.

    Reduces model weight memory footprint and accelerates CPU matrix multiplications
    using INT8 GEMM operations.

    Args:
        input_model_path: Path to FP32 ONNX model.
        output_model_path: Destination path for quantized ONNX model.
        per_channel: Whether to quantize weights per-channel.
        weight_type: Quantization weight format ('qint8' or 'quint8').

    Returns:
        Path to quantized artifact.
    """
    src_path = Path(input_model_path)
    dst_path = Path(output_model_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {src_path}")

    q_type = QuantType.QInt8 if weight_type.lower() == "qint8" else QuantType.QUInt8

    try:
        quantize_dynamic(
            model_input=str(src_path),
            model_output=str(dst_path),
            per_channel=per_channel,
            weight_type=q_type,
        )
    except Exception as e:
        logger.error("Quantization failed: %s", e)
        raise ExportError(f"ONNX Dynamic Quantization failed: {e}") from e

    # Structural check on quantized graph
    q_model = onnx.load(str(dst_path))
    onnx.checker.check_model(q_model)

    # Save quantization metadata
    meta = {
        "source_fp32_model": src_path.name,
        "quantized_target": dst_path.name,
        "weight_type": str(q_type),
        "per_channel": per_channel,
        "source_size_bytes": src_path.stat().st_size,
        "quantized_size_bytes": dst_path.stat().st_size,
        "size_reduction_pct": float(
            (1.0 - (dst_path.stat().st_size / max(1, src_path.stat().st_size))) * 100.0
        ),
    }
    save_json(meta, dst_path.parent / f"{dst_path.stem}_quant_meta.json")
    logger.info(
        "Successfully quantized %s -> %s (Size reduction: %.1f%%)",
        src_path.name,
        dst_path.name,
        meta["size_reduction_pct"],
    )

    return dst_path
