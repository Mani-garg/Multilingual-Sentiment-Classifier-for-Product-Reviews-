# Model Evaluation & Multilingual Performance Report
**Generated:** 2026-10-26 04:51:42 UTC  
**Dataset:** Multilingual Product Reviews  
**Primary Metric:** Macro-F1 (unweighted arithmetic mean across classes)  

---

## 1. Global Model Comparison

| Architecture / Model | Macro-F1 (%) | Accuracy (%) | Weighted-F1 (%) | Avg Latency (ms) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression Baseline** | 100.00% | 100.00% | 100.00% | 0.01 ms | Verified |
| **Multilingual Transformer (DistilBERT)** | *N/A* | *N/A* | *N/A* | *N/A* | Pending Run |
| **ONNX Runtime Optimized Model** | *N/A* | *N/A* | *N/A* | *N/A* | Pending Run |

## 2. Per-Class Sentiment Performance

| Class | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| **Negative** | 1.0000 | 1.0000 | 1.0000 |
| **Neutral** | 1.0000 | 1.0000 | 1.0000 |
| **Positive** | 1.0000 | 1.0000 | 1.0000 |

## 3. Multilingual Evaluation Breakdown

| Language | Code | Test Samples | Accuracy (%) | Macro-F1 (%) | Error Share (%) | Statistical Validity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| DE | `de` | 22 | 100.0% | 100.0% | 0.0% | Sufficient (>=10) |
| EN | `en` | 27 | 100.0% | 100.0% | 0.0% | Sufficient (>=10) |
| ES | `es` | 22 | 100.0% | 100.0% | 0.0% | Sufficient (>=10) |
| FR | `fr` | 21 | 100.0% | 100.0% | 0.0% | Sufficient (>=10) |
| JA | `ja` | 17 | 100.0% | 100.0% | 0.0% | Sufficient (>=10) |
| ZH | `zh` | 17 | 100.0% | 100.0% | 0.0% | Sufficient (>=10) |
