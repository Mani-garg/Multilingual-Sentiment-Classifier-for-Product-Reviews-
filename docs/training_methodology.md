# Training Methodology

## 1. Dual-Model Strategy

To guarantee rapid development velocity and establish an honest, uninflated reference performance, our system implements a **dual-model strategy**:
1. **TF-IDF + Logistic Regression Baseline**:
   - Feature extractor: Scikit-learn `TfidfVectorizer` with sublinear TF scaling, n-gram range `(1, 2)`, character-level n-gram fallbacks for CJK scripts (`char_wb`, `(2, 5)`), and max features of 25,000.
   - Classifier: `LogisticRegression` with L2 regularization (`C=1.0`), balanced class weights (`class_weight="balanced"`), and L-BFGS solver.
   - Purpose: Near-instantaneous training (< 5 seconds), zero GPU dependency, rock-solid determinism, and minimal inference latency (< 1 ms).
2. **Fine-Tuned Multilingual Transformer**:
   - Architecture: Multilingual encoder checkpoint (`google-bert/bert-base-multilingual-cased` or `xlm-roberta-base`).
   - Purpose: Rich cross-lingual transfer, subword context modeling, and high semantic fidelity on complex linguistic structures (negation, sarcasm, mixed sentiments).

---

## 2. Dynamic Padding & Collator Design

In production NLP workflows, static sequence padding (padding every sample to the absolute maximum dataset length, e.g., 512 tokens) is highly wasteful:
- If a mini-batch has an average sequence length of 48 tokens, padding to 512 wastes over 90% of tensor computations on pad mask operations.
- **Dynamic Padding Collator**: Batches are grouped by length, and sequences within a mini-batch are padded *only* to the longest sequence in that specific mini-batch.
- Result: 2.5x to 4x throughput acceleration during training and evaluation on variable-length consumer reviews.

---

## 3. Loss Functions & Class Imbalance

Real-world product review datasets exhibit inherent class imbalance (positive reviews often outnumber neutral reviews 4-to-1). The pipeline supports:
- **Weighted Categorical Cross-Entropy**:
  $$\mathcal{L} = -\sum_{c=1}^C w_c y_c \log(\hat{y}_c)$$
  where weights $w_c = \frac{N}{C \cdot N_c}$ counterbalance majority classes.
- **Focal Loss**:
  $$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  with focusing parameter $\gamma = 2.0$, down-weighting easy examples and forcing gradient updates on ambiguous, neutral, or hard boundary samples.

---

## 4. Training-Only Targeted Augmentation Protocol

To combat systematic misclassification of negation particles (e.g., *"not bad"*, *"never fails"*):
1. **Isolation**: Augmentations are executed strictly on the training partition. The validation and test partitions remain untouched.
2. **Rule-Based Negation Injection**:
   - Matches negation operators across English, German, French, and Spanish.
   - Generates contrastive pairs and updates ground-truth sentiment labels predictably.
3. **Synonym & Adverb Perturbation**:
   - Modifies degree adverbs (*"very good"* $\to$ *"extremely good"*, *"slightly bad"*) without flipping sentiment polarity.
4. **Honest Reporting**:
   - Pre-augmentation and post-augmentation test splits are evaluated under identical configurations.
   - If the augmentation yields marginal or negative improvement on a specific distribution, the delta is faithfully documented.
