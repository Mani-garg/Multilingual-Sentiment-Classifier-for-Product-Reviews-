"""Performance and benchmarking routine verification tests."""

import numpy as np

from sentiment_classifier.optimization.compare_artifacts import benchmark_session


def test_benchmark_session_structure() -> None:
    """Benchmark helper must compute correct percentile statistics and throughput."""

    class DummySession:
        def run(self, output_names, feed_dict):
            # Simulate short delay
            return [np.zeros((feed_dict["input_ids"].shape[0], 3))]

    inputs = {"input_ids": np.ones((4, 16))}
    stats = benchmark_session(DummySession(), inputs, num_warmup=2, num_iterations=5)

    assert "median_p50_ms" in stats
    assert "p95_ms" in stats
    assert "p99_ms" in stats
    assert "throughput_samples_per_sec" in stats
    assert stats["batch_size"] == 4.0
