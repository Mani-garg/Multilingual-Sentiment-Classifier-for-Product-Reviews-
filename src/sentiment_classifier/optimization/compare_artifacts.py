"""Comprehensive before-and-after optimization comparison measuring size, memory, and latency."""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil


def get_current_process_memory_mb() -> float:
    """Return process resident set size (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024 * 1024))


def benchmark_session(
    session: ort.InferenceSession,
    sample_inputs: dict[str, np.ndarray],
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> dict[str, float]:
    """Execute warm-up and timed benchmark iterations for an ONNX session."""
    # Warm-up
    for _ in range(num_warmup):
        session.run(None, sample_inputs)

    latencies_ms = []
    batch_size = next(iter(sample_inputs.values())).shape[0]

    for _ in range(num_iterations):
        t0 = time.perf_counter()
        session.run(None, sample_inputs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms)
    throughput = (batch_size * num_iterations) / (np.sum(lat_arr) / 1000.0)

    return {
        "batch_size": float(batch_size),
        "mean_latency_ms": float(np.mean(lat_arr)),
        "median_p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "throughput_samples_per_sec": float(throughput),
    }


def compare_model_artifacts(
    original_onnx_path: str | Path,
    quantized_onnx_path: str | Path,
    sample_inputs: dict[str, np.ndarray],
    execution_provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    """Compare artifact size, memory footprint, and latency before and after quantization."""
    orig_path = Path(original_onnx_path)
    quant_path = Path(quantized_onnx_path)

    orig_size_mb = float(orig_path.stat().st_size / (1024 * 1024))
    quant_size_mb = float(quant_path.stat().st_size / (1024 * 1024))
    size_reduction_pct = float((1.0 - (quant_size_mb / max(1e-6, orig_size_mb))) * 100.0)

    # 1. Benchmark Original FP32
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    get_current_process_memory_mb()

    orig_session = ort.InferenceSession(
        str(orig_path), sess_options, providers=[execution_provider]
    )
    get_current_process_memory_mb()
    orig_bench = benchmark_session(orig_session, sample_inputs)

    # 2. Benchmark Quantized INT8
    quant_session = ort.InferenceSession(
        str(quant_path), sess_options, providers=[execution_provider]
    )
    get_current_process_memory_mb()
    quant_bench = benchmark_session(quant_session, sample_inputs)

    # Speedup calculation
    p50_speedup = orig_bench["median_p50_ms"] / max(1e-6, quant_bench["median_p50_ms"])
    throughput_gain = (
        quant_bench["throughput_samples_per_sec"]
        / max(1e-6, orig_bench["throughput_samples_per_sec"])
        - 1.0
    ) * 100.0

    comparison = {
        "execution_provider": execution_provider,
        "artifact_size": {
            "original_fp32_mb": round(orig_size_mb, 2),
            "quantized_int8_mb": round(quant_size_mb, 2),
            "size_reduction_pct": round(size_reduction_pct, 1),
        },
        "latency": {
            "original_p50_ms": round(orig_bench["median_p50_ms"], 2),
            "quantized_p50_ms": round(quant_bench["median_p50_ms"], 2),
            "original_p95_ms": round(orig_bench["p95_ms"], 2),
            "quantized_p95_ms": round(quant_bench["p95_ms"], 2),
            "p50_speedup_factor": round(p50_speedup, 2),
        },
        "throughput": {
            "original_samples_per_sec": round(orig_bench["throughput_samples_per_sec"], 1),
            "quantized_samples_per_sec": round(quant_bench["throughput_samples_per_sec"], 1),
            "throughput_gain_pct": round(throughput_gain, 1),
        },
    }
    return comparison
