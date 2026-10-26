# Dataset Card: Multilingual Product Reviews

## 1. Dataset Summary

The Multilingual Sentiment Classifier dataset comprises product reviews across six primary languages: English (`en`), German (`de`), Spanish (`es`), French (`fr`), Japanese (`ja`), and Chinese (`zh`). Each review is annotated with a 3-class sentiment label (`negative`, `neutral`, `positive`), a normalized star rating (1.0 to 5.0), language code, review identifier, and grouping key for product-level leakage prevention.

### Supported Tasks & Leaderboards
- Multi-class text classification (Sentiment Analysis).
- Cross-lingual zero-shot and few-shot sentiment transfer.
- Robustness evaluation across linguistic failure modes (negation, contrastive discourse, mixed sentiment).

---

## 2. Dataset Structure

### Data Instances
```json
{
  "review_id": "rev_000042",
  "review_text": "The battery life is phenomenal and build quality is top-notch! Product ID #42.",
  "language": "en",
  "rating": 5.0,
  "sentiment_label": "positive",
  "source_dataset": "demo_multilingual_reviews",
  "source_record_id": "rec_7a3bc91e",
  "group_id": "PROD_0012"
}
```

### Data Fields
- `review_id` *(string)*: Unique identifier for each review record.
- `review_text` *(string)*: Raw text containing consumer product feedback.
- `language` *(string, ISO 639-1)*: Two-letter ISO language code (`en`, `de`, `es`, `fr`, `ja`, `zh`).
- `rating` *(float, optional)*: Rating on a 1.0 - 5.0 scale.
- `sentiment_label` *(string)*: Categorical ground-truth sentiment: `negative`, `neutral`, `positive`.
- `source_dataset` *(string)*: Source provenance (e.g. `amazon_multilingual`, `demo_multilingual_reviews`).
- `source_record_id` *(string)*: Original upstream record identifier.
- `group_id` *(string, optional)*: Grouping key (e.g., product ASIN/ID or reviewer ID) utilized for cluster-aware cross-validation.

---

## 3. Data Splits & Leakage Prevention

The dataset enforces multi-key stratified splitting:
- **80% Train**: Model parameter optimization and training-only targeted augmentation.
- **10% Validation**: Early stopping, learning rate scheduling, and hyperparameter tuning.
- **10% Test**: Held-out final performance evaluation and honest error analysis.

### Stratification & Deduplication Rules
1. Exact and normalized near-duplicate texts are detected and dropped prior to splitting.
2. The stratification key combines `sentiment_label` and `language` to preserve class and language balances identically across all partitions.
3. Automated partition overlap tests verify zero review ID or raw text leakage between splits (`assert train_test_overlap == 0`).

---

## 4. Linguistic Characteristics

| Language | ISO Code | Script Family | Typical Token Length (Subwords) | Challenges |
|---|---|---|---|---|
| English | `en` | Latin | 15 - 120 | Colloquial sarcasm, idioms, contracted negations (`won't`, `can't`) |
| German | `de` | Latin | 12 - 95 | Compound nouns (`Akkulaufzeit`), separated prefix verbs (`fällt ... aus`) |
| Spanish | `es` | Latin | 18 - 140 | Double negatives (`no me gustó nada`), regional slang |
| French | `fr` | Latin | 16 - 130 | Bipartite negation (`ne ... pas`), elisions (`d'`, `l'`) |
| Japanese | `ja` | Kanji / Kana | 20 - 160 | Lack of whitespace word boundaries, polite vs blunt forms (`ません`, `ない`) |
| Chinese | `zh` | Simplified Hanzi | 15 - 150 | Character-level semantics, negation particles (`不`, `没`, `毫无`) |
