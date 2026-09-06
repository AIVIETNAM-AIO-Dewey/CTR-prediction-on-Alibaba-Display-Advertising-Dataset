"""Validated probability and threshold metrics for binary CTR evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


def _validate_inputs(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).ravel()
    probabilities = np.asarray(y_prob, dtype=np.float64).ravel()
    if labels.size == 0:
        raise ValueError("Evaluation data is empty.")
    if labels.size != probabilities.size:
        raise ValueError(
            f"Labels and probabilities have different lengths: {labels.size} != {probabilities.size}."
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation labels must contain only binary values 0 and 1.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must lie in [0, 1].")
    if np.unique(labels).size < 2:
        raise ValueError(
            "Evaluation split must contain both classes; ROC-AUC and PR-AUC are undefined otherwise."
        )
    return labels.astype(np.int8, copy=False), probabilities


def compute_probability_metrics(y_true: Any, y_prob: Any) -> Dict[str, float]:
    """Compute threshold-independent binary probability metrics."""
    labels, probabilities = _validate_inputs(y_true, y_prob)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _threshold_row(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, Any]:
    predicted = (probabilities >= threshold).astype(np.int8)
    true_positive = int(((labels == 1) & (predicted == 1)).sum())
    false_positive = int(((labels == 0) & (predicted == 1)).sum())
    true_negative = int(((labels == 0) & (predicted == 0)).sum())
    false_negative = int(((labels == 1) & (predicted == 0)).sum())
    negative_precision = (
        true_negative / (true_negative + false_negative)
        if true_negative + false_negative
        else 0.0
    )
    positive_f1 = float(f1_score(labels, predicted, zero_division=0))
    negative_recall = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": positive_f1,
        "negative_precision": float(negative_precision),
        "negative_recall": float(negative_recall),
        "negative_f1": float(negative_f1),
        "macro_f1": float((positive_f1 + negative_f1) / 2),
        "specificity": float(
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        ),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "positive_rate": float(predicted.mean()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def compute_threshold_metrics(
    y_true: Any,
    y_prob: Any,
    thresholds: Iterable[float],
) -> List[Dict[str, Any]]:
    """Return classification metrics and confusion counts for each threshold."""
    labels, probabilities = _validate_inputs(y_true, y_prob)
    result: List[Dict[str, Any]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must lie in [0, 1], got {threshold}.")
        result.append(_threshold_row(labels, probabilities, threshold))
    if not result:
        raise ValueError("At least one threshold is required.")
    return result


def select_best_f1_threshold(y_true: Any, y_prob: Any) -> float:
    """Select the validation threshold maximizing positive-class F1."""
    labels, probabilities = _validate_inputs(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    best = int(np.flatnonzero(f1_values == f1_values.max())[0])
    return float(thresholds[best])
