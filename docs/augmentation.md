# Targeted Data Augmentation Protocol

## 1. Motivation

Supervised sentiment models routinely fail when encountering negation operators. For example, standard corpora frequently contain many instances of `"good"` (positive) and relatively few instances of `"not good"` (negative). As a consequence, models learn an outsized positive weight for the token `"good"` that overpowers the preceding negation.

Rather than applying arbitrary, unguided data perturbations, our pipeline implements **Targeted Linguistic Augmentation** focused squarely on negation and adverbial modifiers.

---

## 2. Leakage-Free Augmentation Rules

To guarantee academic and production integrity:
1. **Train Split Only**: Augmentations are generated exclusively on `train.jsonl`.
2. **Never Touch Test or Validation Sets**: `val.jsonl` and `test.jsonl` are strictly preserved in their authentic, unaugmented form.
3. **Controlled Generation**: The augmentation ratio is bounded (e.g. 15–25% of training volume) to prevent synthetic drift.
4. **Deterministic Reproducibility**: Random seeds are locked for deterministic generation.

---

## 3. Augmentation Techniques

### Technique A: Controlled Negation Inversion
Replaces positive expressions with negated equivalents (or vice versa) and flips the target ground-truth label:
- `"The display is excellent."` (positive) $\to$ `"The display is not good."` (negative)
- `"Das Gerät ist toll."` (positive) $\to$ `"Das Gerät ist nicht gut."` (negative)
- `"Ce produit fonctionne bien."` (positive) $\to$ `"Ce produit ne fonctionne pas bien."` (negative)

### Technique B: Neutral Adverbial Perturbation
Modifies degree modifiers without altering the underlying polarity label:
- `"Good camera quality."` $\to$ `"Fairly good camera quality."` (positive)
- `"Slow delivery."` $\to$ `"Somewhat slow delivery."` (negative)

---

## 4. Empirical Ablation Protocol

The system runs a controlled ablation via `scripts/run_augmentation_experiment.py`:
1. **Condition A (Control)**: Train model strictly on the unaugmented training partition $D_{\text{train}}$. Evaluate on held-out test partition $D_{\text{test}}$.
2. **Condition B (Augmented)**: Train model on augmented partition $D_{\text{train}} \cup D_{\text{aug}}$. Evaluate on the *exact same* held-out test partition $D_{\text{test}}$.
3. **Statistical Comparison**: Record macro-F1, accuracy, and per-class differences.

Results are written to `artifacts/reports/augmentation_experiment_report.json`.
