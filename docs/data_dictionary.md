# Data Dictionary

## 1. Schema Specifications

All tabular and JSONL datasets managed by the pipeline conform to the strict schema defined below. Schema validation is enforced at ingestion time using Pydantic v2 in `sentiment_classifier.data.validation.ReviewRecord`.

| Column Name | Data Type | Nullable | Valid Values / Constraints | Description |
|---|---|---|---|---|
| `review_id` | String (`str`) | No | Format: `rev_[a-zA-Z0-9_\-]+`, Non-empty | Unique persistent identifier for the review. |
| `review_text` | String (`str`) | No | Min length: 2 characters; Max length: 10,000 characters | Unprocessed or normalized review text. |
| `language` | String (`str`) | No | 2-letter ISO 639-1 code: `en`, `de`, `es`, `fr`, `ja`, `zh` | Primary language identified in the review. |
| `sentiment_label` | String (`str`) | No | `negative`, `neutral`, `positive` | 3-class ground-truth sentiment annotation. |
| `rating` | Float (`float`) | Yes | `1.0 <= rating <= 5.0` | Numerical consumer rating if available. |
| `source_dataset` | String (`str`) | No | Non-empty string (e.g. `amazon_multilingual`) | Data lineage tag indicating source corpus. |
| `source_record_id` | String (`str`) | No | Non-empty string | Record ID in the upstream source dataset. |
| `group_id` | String (`str`) | Yes | String (e.g. `PROD_0042` or `USER_9182`) | Group identifier for group-aware splitting. |

---

## 2. Sentiment Label Mapping

The system natively uses 3 discrete classes. When ingesting raw star-ratings or binary sentiment datasets, the following configurable mappings apply:

```yaml
# 5-star rating conversion
rating_to_sentiment:
  1.0: negative
  2.0: negative
  3.0: neutral
  4.0: positive
  5.0: positive

# 2-class binary conversion
binary_to_sentiment:
  0: negative
  1: positive
```

---

## 3. Preprocessed & Model-Ready Features

When converting text to subword token sequences for transformer training and ONNX execution:

| Feature Name | Type | Dimensions | Description |
|---|---|---|---|
| `input_ids` | `int64` | `[batch_size, seq_len]` | Token IDs produced by the multilingual tokenizer. |
| `attention_mask` | `int64` | `[batch_size, seq_len]` | Binary mask (`1` for real tokens, `0` for padding). |
| `sentiment_label_id` | `int64` | `[batch_size]` | Integer index: `0: negative`, `1: neutral`, `2: positive`. |

Dynamic padding determines `seq_len` on each batch up to `max_length = 256` or `512`.
