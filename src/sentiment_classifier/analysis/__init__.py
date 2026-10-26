"""Evaluation metrics, multilingual language analysis, confusion matrices, and error categorization."""

from sentiment_classifier.analysis.confusion import format_confusion_matrix
from sentiment_classifier.analysis.error_analysis import ErrorAnalyzer, categorize_error
from sentiment_classifier.analysis.language_metrics import compute_multilingual_breakdown
from sentiment_classifier.analysis.metrics import calculate_all_metrics
from sentiment_classifier.analysis.reports import generate_markdown_evaluation_report

__all__ = [
    "calculate_all_metrics",
    "format_confusion_matrix",
    "compute_multilingual_breakdown",
    "ErrorAnalyzer",
    "categorize_error",
    "generate_markdown_evaluation_report",
]
