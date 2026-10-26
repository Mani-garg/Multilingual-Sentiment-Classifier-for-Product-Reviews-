"""Automated numerical parity validation comparing reference model and ONNX predictions."""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class ParityReport:
    """Report detailing numerical divergence between reference and optimized graphs."""

    sample_count: int
    label_agreement_rate: float
    max_absolute_error: float
    mean_absolute_error: float
    passed_tolerance: bool
    tolerance_atol: float
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_numerical_parity(
    reference_logits: np.ndarray,
    onnx_session: ort.InferenceSession,
    sample_inputs: dict[str, np.ndarray],
    atol: float = 1e-4,
    min_agreement: float = 0.98,
) -> ParityReport:
    """Validate numerical parity between reference logits and ONNX session outputs.

    Args:
        reference_logits: Float array of shape [batch_size, num_classes].
        onnx_session: Initialized onnxruntime.InferenceSession.
        sample_inputs: Dictionary mapping input names ('input_ids', 'attention_mask') to numpy arrays.
        atol: Absolute numerical tolerance.
        min_agreement: Minimum fraction of identical argmax class predictions required.

    Returns:
        ParityReport dataclass.
    """
    # Execute ONNX forward pass
    ort_outputs = onnx_session.run(None, sample_inputs)
    onnx_logits = ort_outputs[0]

    if reference_logits.shape != onnx_logits.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference_logits.shape} vs ONNX {onnx_logits.shape}"
        )

    # Compute numerical errors
    abs_diff = np.abs(reference_logits - onnx_logits)
    max_error = float(np.max(abs_diff))
    mean_error = float(np.mean(abs_diff))

    # Compute label agreement
    ref_preds = np.argmax(reference_logits, axis=-1)
    onnx_preds = np.argmax(onnx_logits, axis=-1)
    agreements = (ref_preds == onnx_preds).astype(float)
    agreement_rate = float(np.mean(agreements))

    passed = (max_error <= atol) and (agreement_rate >= min_agreement)

    details = []
    for i in range(min(5, len(ref_preds))):
        details.append(
            {
                "sample_index": i,
                "ref_pred": int(ref_preds[i]),
                "onnx_pred": int(onnx_preds[i]),
                "ref_logits": [round(float(v), 4) for v in reference_logits[i]],
                "onnx_logits": [round(float(v), 4) for v in onnx_logits[i]],
                "max_diff": round(float(np.max(abs_diff[i])), 5),
            }
        )

    report = ParityReport(
        sample_count=len(ref_preds),
        label_agreement_rate=agreement_rate,
        max_absolute_error=max_error,
        mean_absolute_error=mean_error,
        passed_tolerance=passed,
        tolerance_atol=atol,
        details=details,
    )

    logger.info(
        "Parity Check | Agreement: %.2f%% | MaxAE: %.5f | MeanAE: %.5f | Passed: %s",
        agreement_rate * 100.0,
        max_error,
        mean_error,
        passed,
    )
    return report
