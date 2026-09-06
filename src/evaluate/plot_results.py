"""Headless ROC, PR and calibration plots for model comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate.evaluator import ModelEvaluationResult


def _plot_split(
    results: Sequence[ModelEvaluationResult],
    split: str,
    output_dir: Path,
) -> List[Path]:
    split_results = [
        (result.model_key, getattr(result, split))
        for result in results
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    roc_path = output_dir / f"{split}_roc.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_key, evaluation in split_results:
        ax.plot(
            evaluation.curves["roc_fpr"],
            evaluation.curves["roc_tpr"],
            label=f"{model_key} (AUC={evaluation.probability_metrics['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(title=f"{split.title()} ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    paths.append(roc_path)

    pr_path = output_dir / f"{split}_precision_recall.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_key, evaluation in split_results:
        ax.plot(
            evaluation.curves["pr_recall"],
            evaluation.curves["pr_precision"],
            label=f"{model_key} (AP={evaluation.probability_metrics['pr_auc']:.4f})",
        )
    ax.set(
        title=f"{split.title()} Precision–Recall curves",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    paths.append(pr_path)

    calibration_path = output_dir / f"{split}_calibration.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_key, evaluation in split_results:
        ax.plot(
            evaluation.curves["calibration_predicted"],
            evaluation.curves["calibration_true"],
            marker="o",
            label=model_key,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(
        title=f"{split.title()} calibration curves",
        xlabel="Mean predicted probability",
        ylabel="Fraction of positives",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(calibration_path, dpi=150)
    plt.close(fig)
    paths.append(calibration_path)
    return paths


def save_comparison_plots(
    results: Sequence[ModelEvaluationResult],
    output_dir: str | Path,
) -> List[Path]:
    """Save ROC, precision–recall and calibration plots for both splits."""
    destination = Path(output_dir)
    paths: List[Path] = []
    for split in ("validation", "test"):
        paths.extend(_plot_split(results, split, destination))
    return paths
