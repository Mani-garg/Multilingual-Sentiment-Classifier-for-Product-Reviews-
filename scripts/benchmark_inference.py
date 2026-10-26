"""Inference latency and throughput benchmarking suite."""

import argparse
import os
import platform
import time

import numpy as np
import psutil

from sentiment_classifier.config import get_settings
from sentiment_classifier.inference.predictor import SentimentPredictor
from sentiment_classifier.utils.serialization import save_json


def benchmark_batch_inference(
    predictor: SentimentPredictor,
    sample_texts: list[str],
    batch_size: int,
    num_warmup: int = 5,
    num_iterations: int = 25,
) -> dict[str, float]:
    """Execute warm-up and timed iterations for a specific batch size."""
    # Build batch
    batch_texts = (sample_texts * ((batch_size // len(sample_texts)) + 1))[:batch_size]

    # Warmup
    for _ in range(num_warmup):
        predictor.predict_batch(batch_texts)

    latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        predictor.predict_batch(batch_texts)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms)
    total_time_s = np.sum(lat_arr) / 1000.0
    throughput = (batch_size * num_iterations) / max(1e-6, total_time_s)

    return {
        "batch_size": batch_size,
        "mean_latency_ms": round(float(np.mean(lat_arr)), 2),
        "median_p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "latency_per_sample_ms": round(float(np.mean(lat_arr) / batch_size), 3),
        "throughput_samples_per_sec": round(float(throughput), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency and throughput.")
    parser.add_argument(
        "--backend", type=str, default="baseline", choices=["baseline", "onnx", "quantized_onnx"]
    )
    parser.add_argument("--batch-sizes", type=str, default="1,8,32")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output", type=str, default="artifacts/reports/inference_benchmark_report.json"
    )
    args = parser.parse_args()

    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]

    # Collect representative multilingual test samples
    samples = [
        "The battery life is phenomenal and build quality is top-notch!",
        "Hervorragende Qualität und super schneller Versand! Sehr zu empfehlen.",
        "Produit d'excellente qualité, finition impeccable et livraison rapide !",
        "¡Excelente producto! Calidad de construcción insuperable y llegó a tiempo.",
        "画質が非常に美しく、操作も直感的で素晴らしい製品です！",
        "屏幕显示效果惊艳，系统流畅无卡顿，性价比极高！",
        "Stopped working after just two days. Complete waste of money.",
        "Völliger Schrott! Ging nach einer Woche kaputt und lässt sich nicht laden.",
    ]

    settings = get_settings()
    settings.inference_backend = args.backend

    # Cold start measurement
    cold_t0 = time.perf_counter()
    predictor = SentimentPredictor(settings=settings)
    cold_start_ms = (time.perf_counter() - cold_t0) * 1000.0

    print(f"Benchmarking Backend: {args.backend} | Cold Start: {cold_start_ms:.2f} ms")

    results_by_batch = []
    for b_size in batch_sizes:
        print(f"Testing batch_size={b_size} ({args.iterations} iterations)...")
        res = benchmark_batch_inference(
            predictor=predictor,
            sample_texts=samples,
            batch_size=b_size,
            num_iterations=args.iterations,
        )
        results_by_batch.append(res)
        print(
            f"  -> p50: {res['median_p50_ms']} ms | p95: {res['p95_ms']} ms | Throughput: {res['throughput_samples_per_sec']} samples/s"
        )

    # Hardware & Environment metadata
    process = psutil.Process(os.getpid())
    memory_rss_mb = process.memory_info().rss / (1024 * 1024)

    benchmark_record = {
        "hardware_and_environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_architecture": platform.machine(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version(),
        },
        "benchmark_config": {
            "backend": args.backend,
            "cold_start_ms": round(cold_start_ms, 2),
            "iterations_per_batch": args.iterations,
            "memory_rss_mb": round(memory_rss_mb, 2),
        },
        "batch_benchmarks": results_by_batch,
    }

    save_json(benchmark_record, args.output)
    print(f"\nInference benchmark report saved to: {args.output}")


if __name__ == "__main__":
    main()
