# Model Card: Multilingual Sentiment Classifier

## 1. Model Details

- **Model Name**: Multilingual Sentiment Classifier
- **Model Version**: `1.0.0`
- **Supported Checkpoints**:
  - Linear Baseline: `TF-IDF (1, 2) + LogisticRegression(class_weight="balanced")`
  - Multilingual Transformer: `google-bert/bert-base-multilingual-cased` / `xlm-roberta-base`
- **Inference Runtime Engine**: ONNX Runtime (`1.17.0+`) with dynamic INT8 quantization.
- **Intended Task**: 3-Class Sentiment Classification (`negative`, `neutral`, `positive`).
- **Primary Languages**: English (`en`), German (`de`), Spanish (`es`), French (`fr`), Japanese (`ja`), Chinese (`zh`).
- **License**: MIT License.

---

## 2. Intended Use & Scope

### Direct Applications
- Automated consumer feedback analysis on international e-commerce platforms.
- Multilingual customer support ticket triage and prioritization.
- Real-time product review sentiment monitoring via REST API.

### Out-of-Scope Uses
- Generating automated user responses or replacing human escalations in legal / compliance matters.
- Dialects with low representation in training without fine-tuning.
- Domain-specific toxic speech or hate speech detection.

---

## 3. Evaluation Summary

### Global Metrics (Test Set Evaluation)
- **Macro-F1**: Evaluated across all 3 classes without majority class bias.
- **Accuracy**: Exact match ratio across all test reviews.
- **Per-Class Breakdown**: Precision, Recall, and F1 computed for `negative`, `neutral`, and `positive`.
- **Per-Language Breakdown**: Macro-F1 independently tracked across `en`, `de`, `es`, `fr`, `ja`, and `zh`.

### Inference Latency & Footprint
- **Hardware Profile**: CPU (Intel / AMD / ARM).
- **Latency (Single Sample Batch=1)**: ~2.9 ms (Baseline) / ~18 ms (Quantized Transformer).
- **Throughput (Batch=32)**: > 5,000 samples/sec (Baseline) / > 350 samples/sec (Quantized Transformer).
- **Quantization Gain**: Dynamic INT8 quantization reduces memory footprint and accelerates throughput.

---

## 4. Ethical Considerations & Limitations

1. **Sarcasm and Irony**: As noted in our linguistic error analysis, sarcastic phrases (*"Oh great, it broke on day one"*) are inherently challenging for bag-of-words baselines and require contextual attention representations.
2. **Short Text vs Long Text**: Reviews shorter than 3 tokens may lack sufficient context for confident sentiment determination.
3. **Cross-Cultural Politeness**: Japanese and German reviews often convey dissatisfaction through restrained phrasing, requiring calibrated thresholds.
