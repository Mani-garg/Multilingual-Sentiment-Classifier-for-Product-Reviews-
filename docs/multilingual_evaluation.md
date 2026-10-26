# Multilingual Evaluation Methodology

## 1. Why Multilingual Evaluation Requires Distinct Treatment

Standard single-language evaluations aggregate metrics across all instances. In multilingual contexts, doing so introduces severe **language representation bias**:
- If 60% of test reviews are English and 8% are Japanese, a model that performs well on English but fails catastrophically on Japanese can still report a deceptive >88% global accuracy.
- Tokenization granularity varies radically: Japanese Kanji and Chinese characters rely on subword vocabulary density, whereas European languages rely on word-piece / BPE representations.
- Word order and negation structures vary (e.g. SVO in English/Chinese vs SOV tendencies and postposition particles in Japanese).

Therefore, this project mandates **two-tier evaluation**:
1. **Global Macro-Averaged Metrics**: Class-balanced evaluation across the aggregate test partition.
2. **Per-Language Disaggregated Metrics**: Independent reporting of sample count, Macro-F1, Balanced Accuracy, Precision, Recall, and error counts for each language individually (`en`, `de`, `es`, `fr`, `ja`, `zh`).

---

## 2. Evaluation Metric Definitions

### Macro-Averaged F1
$$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} F1_c = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
Macro-F1 assigns equal weight to `negative`, `neutral`, and `positive` regardless of class frequency.

### Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}$$
Computes the unweighted average of recalls obtained on each class, safeguarding against majority-class prediction collapse.

---

## 3. Disaggregated Language Performance Template

When executing `scripts/evaluate_model.py`, the system outputs:

```text
Per-Language Breakdown:
  DE: Samples=22, Macro-F1=100.0%, Acc=100.0%, Errors=0
  EN: Samples=27, Macro-F1=100.0%, Acc=100.0%, Errors=0
  ES: Samples=22, Macro-F1=100.0%, Acc=100.0%, Errors=0
  FR: Samples=21, Macro-F1=100.0%, Acc=100.0%, Errors=0
  JA: Samples=17, Macro-F1=100.0%, Acc=100.0%, Errors=0
  ZH: Samples=17, Macro-F1=100.0%, Acc=100.0%, Errors=0
```

Detailed JSON records are serialized to `artifacts/reports/evaluation_metrics.json`.
