"""End-to-end demonstration workflow exercising data, training, optimization, and API serving."""

import os
import subprocess
import sys
import time


def run_step(step_name: str, cmd: list[str]) -> None:
    print("\n==================================================")
    print(f">> STEP: {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print("==================================================")
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, check=True, env=env)
    t1 = time.perf_counter()
    print(f"[OK] Completed in {t1 - t0:.2f} s")


def main() -> None:
    py = sys.executable
    print("==================================================")
    print("  MULTILINGUAL SENTIMENT CLASSIFIER DEMO")
    print("==================================================")

    # 1. Generate sample data
    run_step(
        "Generate Multilingual Sample Data",
        [
            py,
            "scripts/generate_demo_data.py",
            "--output",
            "data/sample/multilingual_reviews_sample.jsonl",
            "--samples",
            "1500",
        ],
    )

    # 2. Ingest, validate, clean, deduplicate, and split
    run_step(
        "Ingest, Validate & Split Dataset",
        [
            py,
            "scripts/download_or_prepare_data.py",
            "--input",
            "data/sample/multilingual_reviews_sample.jsonl",
        ],
    )

    # 3. Train TF-IDF baseline
    run_step(
        "Train TF-IDF + Logistic Regression Baseline",
        [py, "scripts/train_baseline.py"],
    )

    # 4. Run targeted augmentation experiment
    run_step(
        "Run Targeted Augmentation Experiment (Train-Only)",
        [py, "scripts/run_augmentation_experiment.py"],
    )

    # 5. Export and quantize ONNX model
    run_step(
        "Export and Quantize ONNX Runtime Model",
        [py, "scripts/export_onnx.py", "--quantize"],
    )

    # 6. Evaluate and generate markdown report
    run_step(
        "Evaluate Model & Generate Markdown Report",
        [py, "scripts/evaluate_model.py", "--backend", "baseline"],
    )

    # 7. Analyze Linguistic Errors
    run_step(
        "Analyze Linguistic Errors",
        [py, "scripts/analyze_errors.py", "--backend", "baseline"],
    )

    # 8. Benchmark Inference Latency & Throughput
    run_step(
        "Benchmark Inference Latency",
        [py, "scripts/benchmark_inference.py", "--backend", "baseline", "--batch-sizes", "1,8,32"],
    )

    print("\n==================================================")
    print("  DEMO SUITE COMPLETED SUCCESSFULLY!")
    print("  Artifacts generated:")
    print("   - data/processed/ (train.jsonl, val.jsonl, test.jsonl)")
    print("   - artifacts/baseline/ (vectorizer.joblib, classifier.joblib)")
    print("   - artifacts/onnx/ (model.onnx, model.quant.onnx)")
    print(
        "   - artifacts/reports/ (evaluation_report.md, data_quality_report.json, error_analysis_report.json)"
    )
    print("==================================================")


if __name__ == "__main__":
    main()
