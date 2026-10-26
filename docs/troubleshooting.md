# Troubleshooting Guide

## 1. Common Issues and Resolutions

### Issue 1: `AttributeError: _ARRAY_API not found` or NumPy 2.x Incompatibility
- **Cause**: TensorFlow 2.16 / 2.17 C++ wrappers were compiled against NumPy 1.x. Installing unconstrained NumPy 2.x causes a C-API import failure.
- **Resolution**: Pin NumPy to `<2.0.0`:
  ```bash
  uv pip install "numpy<2.0.0"
  ```

---

### Issue 2: Windows DLL Load Failure (`_pywrap_tensorflow_lite_metrics_wrapper` or `torch_cpu.dll`)
- **Cause**: On Windows with Python 3.13, TensorFlow wheels lack pre-built binary support, or PyTorch is missing the Microsoft Visual C++ 2015–2022 Redistributable.
- **Resolution**:
  1. Use Python 3.11: `uv venv .venv --python 3.11`.
  2. For inference, use the ONNX Runtime backend (`BACKEND=onnx` or `BACKEND=baseline`), which runs cleanly on Windows without PyTorch or TensorFlow C++ dependencies.

---

### Issue 3: Stratified K-Fold / Train-Test Split `ValueError: test_size < n_classes`
- **Cause**: When running on very small test datasets where `n_samples * test_size < n_languages * n_classes`, multi-key compound stratification cannot allocate at least one sample per stratum.
- **Resolution**: The pipeline automatically detects this condition and gracefully degrades compound stratification to single-key sentiment stratification (`sentiment_label`).

---

### Issue 4: Dynamic INT8 Quantization Shape Error
- **Cause**: Graph inputs to Gemm/MatMul operations having dynamic 1D tensor rank instead of 2D `[batch, channels]`.
- **Resolution**: The fallback graph generator inserts an explicit `Flatten(axis=1)` node prior to Gemm operations, ensuring strict 2D output compatibility.

---

### Issue 5: Port 8000 Already Bound
- **Cause**: A previous Uvicorn process is still listening on port 8000.
- **Resolution**: Kill existing process or specify an alternate port:
  ```bash
  uvicorn sentiment_classifier.api.app:app --port 8080
  ```
