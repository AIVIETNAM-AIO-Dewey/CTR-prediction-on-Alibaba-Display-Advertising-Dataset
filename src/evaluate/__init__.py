"""Evaluation and benchmark utilities for fitted CTR model artifacts."""

from src.evaluate.evaluator import (
    EvaluationResult,
    ModelEvaluationResult,
    evaluate_artifact,
    evaluate_predictions,
    evaluate_suite,
    load_evaluation_results,
    write_evaluation_outputs,
)
from src.evaluate.metrics import (
    compute_probability_metrics,
    compute_threshold_metrics,
    select_best_f1_threshold,
)

__all__ = [
    "EvaluationResult",
    "ModelEvaluationResult",
    "compute_probability_metrics",
    "compute_threshold_metrics",
    "select_best_f1_threshold",
    "evaluate_predictions",
    "evaluate_artifact",
    "evaluate_suite",
    "write_evaluation_outputs",
    "load_evaluation_results",
]
