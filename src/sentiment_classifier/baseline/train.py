"""Baseline training pipeline: TF-IDF Vectorization with Logistic Regression."""

import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sentiment_classifier.baseline.artifacts import BaselineArtifacts
from sentiment_classifier.labels import ID_TO_LABEL, LABEL_TO_ID, SUPPORTED_LABELS
from sentiment_classifier.preprocessing.text_normalization import normalize_for_baseline


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive classification metrics including per-language breakdowns."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    # Per-class metrics
    class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    class_prec = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    class_rec = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    per_class_metrics = {}
    for idx, label_name in ID_TO_LABEL.items():
        per_class_metrics[label_name] = {
            "f1": float(class_f1[idx]),
            "precision": float(class_prec[idx]),
            "recall": float(class_rec[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # Per-language metrics
    per_language_metrics = {}
    if languages is not None and len(languages) == len(y_true):
        df_lang = pd.DataFrame({"true": y_true, "pred": y_pred, "lang": languages})
        total_errors = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt != yp)

        for lang, group in df_lang.groupby("lang"):
            g_true = group["true"].tolist()
            g_pred = group["pred"].tolist()
            g_count = len(g_true)
            g_errors = sum(1 for yt, yp in zip(g_true, g_pred, strict=False) if yt != yp)

            if g_count >= 5:
                lang_f1 = float(f1_score(g_true, g_pred, average="macro", zero_division=0))
                lang_acc = float(accuracy_score(g_true, g_pred))
            else:
                lang_f1 = None
                lang_acc = None

            per_language_metrics[lang] = {
                "sample_count": g_count,
                "accuracy": lang_acc,
                "macro_f1": lang_f1,
                "error_count": g_errors,
                "error_percentage_of_total": float(g_errors / total_errors * 100)
                if total_errors > 0
                else 0.0,
            }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "labels_order": SUPPORTED_LABELS,
        "per_language": per_language_metrics,
    }


def train_baseline_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifact_dir: str | Path,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 25000,
    min_df: int = 2,
    c_regularization: float = 1.0,
    class_weight: str | None = "balanced",
    random_seed: int = 42,
) -> BaselineArtifacts:
    """Train TF-IDF + Logistic Regression baseline on train_df and evaluate on test_df."""
    start_time = time.perf_counter()

    # Preprocess text
    train_texts = [normalize_for_baseline(t) for t in train_df["review_text"]]
    test_texts = [normalize_for_baseline(t) for t in test_df["review_text"]]

    train_labels = [LABEL_TO_ID[lbl] for lbl in train_df["sentiment_label"]]
    test_labels = [LABEL_TO_ID[lbl] for lbl in test_df["sentiment_label"]]
    test_languages = test_df["language"].tolist() if "language" in test_df.columns else None

    # Fit vectorizer strictly on training data
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    # Fit classifier
    classifier = LogisticRegression(
        C=c_regularization,
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_seed,
        solver="lbfgs",
    )
    classifier.fit(x_train, train_labels)

    training_duration_s = time.perf_counter() - start_time

    # Predict on test
    inf_start = time.perf_counter()
    test_preds = classifier.predict(x_test).tolist()
    inference_duration_ms = (time.perf_counter() - inf_start) * 1000.0 / max(1, len(test_preds))

    metrics = evaluate_predictions(test_labels, test_preds, languages=test_languages)
    metrics["training_duration_seconds"] = float(training_duration_s)
    metrics["avg_inference_latency_ms"] = float(inference_duration_ms)
    metrics["vocab_size"] = len(vectorizer.vocabulary_)

    config = {
        "ngram_range": list(ngram_range),
        "max_features": max_features,
        "min_df": min_df,
        "c_regularization": c_regularization,
        "class_weight": class_weight,
        "random_seed": random_seed,
    }

    artifacts = BaselineArtifacts(
        vectorizer=vectorizer,
        classifier=classifier,
        label_to_id=LABEL_TO_ID,
        id_to_label=ID_TO_LABEL,
        config=config,
        metrics=metrics,
    )

    artifacts.save(artifact_dir)
    return artifacts
