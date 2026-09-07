"""Artifact evaluation orchestration and machine-readable result writers."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.models.data_utils import (
    ctr_partition_rows,
    iter_ctr_partition_batches,
    load_ctr_partition,
)
from src.models.train import get_model_class, load_config
from src.evaluate.metrics import (
    compute_probability_metrics,
    compute_threshold_metrics,
    select_best_f1_threshold,
)


def _downsample(values: np.ndarray, max_points: int = 2_000) -> List[float]:
    if values.size <= max_points:
        return values.astype(float).tolist()
    indices = np.linspace(0, values.size - 1, max_points, dtype=np.int64)
    return values[indices].astype(float).tolist()


def _curve_data(y_true: Any, y_prob: Any) -> Dict[str, List[float]]:
    labels = np.asarray(y_true).ravel()
    probabilities = np.asarray(y_prob, dtype=np.float64).ravel()
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    calibration_true, calibration_pred = calibration_curve(
        labels, probabilities, n_bins=10, strategy="quantile"
    )
    return {
        "roc_fpr": _downsample(fpr),
        "roc_tpr": _downsample(tpr),
        "pr_recall": _downsample(recall),
        "pr_precision": _downsample(precision),
        "calibration_predicted": calibration_pred.astype(float).tolist(),
        "calibration_true": calibration_true.astype(float).tolist(),
    }


@dataclass
class EvaluationResult:
    split: str
    n_rows: int
    positive_rate: float
    probability_metrics: Dict[str, float]
    threshold_metrics: List[Dict[str, Any]]
    selected_threshold: float
    curves: Dict[str, List[float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationResult":
        return cls(
            split=str(payload["split"]),
            n_rows=int(payload["n_rows"]),
            positive_rate=float(payload["positive_rate"]),
            probability_metrics={
                str(key): float(value)
                for key, value in dict(payload["probability_metrics"]).items()
            },
            threshold_metrics=[dict(row) for row in payload["threshold_metrics"]],
            selected_threshold=float(payload["selected_threshold"]),
            curves={
                str(key): [float(value) for value in values]
                for key, values in dict(payload.get("curves", {})).items()
            },
        )


@dataclass
class ModelEvaluationResult:
    model_key: str
    artifact_path: str
    validation: EvaluationResult
    test: EvaluationResult
    manifest_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelEvaluationResult":
        return cls(
            model_key=str(payload["model_key"]),
            artifact_path=str(payload.get("artifact_path", "")),
            validation=EvaluationResult.from_dict(payload["validation"]),
            test=EvaluationResult.from_dict(payload["test"]),
            manifest_path=payload.get("manifest_path"),
            metadata=dict(payload.get("metadata", {})),
        )


def evaluate_predictions(
    model_key: str,
    y_val: Any,
    p_val: Any,
    y_test: Any,
    p_test: Any,
    thresholds: Iterable[float],
    artifact_path: str = "",
    manifest_path: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ModelEvaluationResult:
    """Evaluate one model's validation/test probabilities with one frozen threshold."""
    val_labels = np.asarray(y_val).ravel()
    val_probabilities = np.asarray(p_val, dtype=np.float64).ravel()
    test_labels = np.asarray(y_test).ravel()
    test_probabilities = np.asarray(p_test, dtype=np.float64).ravel()
    selected_threshold = select_best_f1_threshold(val_labels, val_probabilities)
    threshold_values = list(thresholds)
    if not any(np.isclose(selected_threshold, value) for value in threshold_values):
        threshold_values.append(selected_threshold)

    validation_thresholds = compute_threshold_metrics(
        val_labels, val_probabilities, threshold_values
    )
    test_thresholds = compute_threshold_metrics(
        test_labels, test_probabilities, threshold_values
    )
    for row in validation_thresholds:
        row["selected"] = bool(np.isclose(row["threshold"], selected_threshold))
    for row in test_thresholds:
        row["selected"] = bool(np.isclose(row["threshold"], selected_threshold))

    validation = EvaluationResult(
        split="validation",
        n_rows=int(val_labels.size),
        positive_rate=float(val_labels.mean()),
        probability_metrics=compute_probability_metrics(val_labels, val_probabilities),
        threshold_metrics=validation_thresholds,
        selected_threshold=float(selected_threshold),
        curves=_curve_data(val_labels, val_probabilities),
    )
    test = EvaluationResult(
        split="test",
        n_rows=int(test_labels.size),
        positive_rate=float(test_labels.mean()),
        probability_metrics=compute_probability_metrics(test_labels, test_probabilities),
        threshold_metrics=test_thresholds,
        selected_threshold=float(selected_threshold),
        curves=_curve_data(test_labels, test_probabilities),
    )
    return ModelEvaluationResult(
        model_key=model_key,
        artifact_path=str(artifact_path),
        validation=validation,
        test=test,
        manifest_path=str(manifest_path) if manifest_path else None,
        metadata=dict(metadata or {}),
    )


def evaluate_artifact(
    model_key: str,
    config: Mapping[str, Any] | str | Path,
    artifact_path: str | Path,
    processed_dir: str | Path,
    thresholds: Iterable[float],
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: int = 42,
    manifest_path: Optional[str | Path] = None,
    batch_size: Optional[int] = None,
) -> ModelEvaluationResult:
    """Load one persisted wrapper and evaluate it on validation and test partitions."""
    cfg = load_config(str(config)) if isinstance(config, (str, Path)) else dict(config)
    model = get_model_class(model_key).load(artifact_path)
    features = list(model.feature_names)
    target = cfg.get("features", {}).get("target", "clk")
    use_fe = cfg.get("data", {}).get("use_fe", True)

    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    if batch_size is not None and (sample_fraction is not None or (sample_size or 0) > 0):
        raise ValueError("Streaming evaluation does not support sampling; use all rows.")

    if batch_size is not None:
        with tempfile.TemporaryDirectory(prefix=f"eval_{model_key}_") as temporary:
            def stream_predictions(split: str) -> tuple[np.memmap, np.memmap]:
                rows = ctr_partition_rows(processed_dir, split, use_fe=use_fe)
                model_features = list(features)
                labels_path = Path(temporary) / f"{split}_labels.dat"
                probabilities_path = Path(temporary) / f"{split}_probabilities.dat"
                labels = np.memmap(labels_path, mode="w+", dtype=np.int8, shape=(rows,))
                probabilities = np.memmap(
                    probabilities_path, mode="w+", dtype=np.float64, shape=(rows,)
                )
                offset = 0
                for batch_features, batch_labels in iter_ctr_partition_batches(
                    processed_dir, split, model_features, target_col=target, use_fe=use_fe,
                    batch_size=batch_size,
                ):
                    batch_probabilities = model.predict_proba(batch_features)
                    end = offset + len(batch_labels)
                    labels[offset:end] = np.asarray(batch_labels, dtype=np.int8)
                    probabilities[offset:end] = np.asarray(batch_probabilities, dtype=np.float64)
                    offset = end
                    del batch_features, batch_labels, batch_probabilities
                if offset != rows:
                    raise RuntimeError(f"{split} streamed {offset} rows; expected {rows}.")
                labels.flush()
                probabilities.flush()
                return labels, probabilities

            y_val, p_val = stream_predictions("val")
            y_test, p_test = stream_predictions("test")
            result = evaluate_predictions(
                model_key=model_key, y_val=y_val, p_val=p_val, y_test=y_test, p_test=p_test,
                thresholds=thresholds, artifact_path=str(artifact_path),
                manifest_path=str(manifest_path) if manifest_path else None,
                metadata={
                    "feature_count": len(features), "use_fe": bool(use_fe),
                    "eval_sample_size": 0, "eval_sample_fraction": None,
                    "random_seed": random_seed, "eval_mode": "streaming",
                    "eval_batch_size": batch_size,
                },
            )
            del y_val, p_val, y_test, p_test
            return result

    X_val, y_val = load_ctr_partition(
        str(processed_dir),
        "val",
        features,
        target_col=target,
        use_fe=use_fe,
        sample_size=sample_size,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
    )
    p_val = model.predict_proba(X_val)
    del X_val

    X_test, y_test = load_ctr_partition(
        str(processed_dir),
        "test",
        features,
        target_col=target,
        use_fe=use_fe,
        sample_size=sample_size,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
    )
    p_test = model.predict_proba(X_test)
    del X_test

    return evaluate_predictions(
        model_key=model_key,
        y_val=y_val,
        p_val=p_val,
        y_test=y_test,
        p_test=p_test,
        thresholds=thresholds,
        artifact_path=str(artifact_path),
        manifest_path=str(manifest_path) if manifest_path else None,
        metadata={
            "feature_count": len(features),
            "use_fe": bool(use_fe),
            "eval_sample_size": sample_size,
            "eval_sample_fraction": sample_fraction,
            "random_seed": random_seed,
            "eval_mode": "in_memory",
        },
    )


def evaluate_suite(
    model_specs: Sequence[Mapping[str, Any]],
    processed_dir: str | Path,
    thresholds: Iterable[float],
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: int = 42,
) -> List[ModelEvaluationResult]:
    """Evaluate model specs sequentially so only one model's data is resident at once."""
    results: List[ModelEvaluationResult] = []
    for spec in model_specs:
        result = evaluate_artifact(
            model_key=str(spec["model_key"]),
            config=spec["config"],
            artifact_path=spec["artifact_path"],
            processed_dir=processed_dir,
            thresholds=thresholds,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
            random_seed=random_seed,
            manifest_path=spec.get("manifest_path"),
        )
        results.append(result)
    return results


def write_evaluation_outputs(
    results: Sequence[ModelEvaluationResult],
    experiments_dir: str | Path,
    plots_dir: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Write JSON/CSV results and comparison plots without persisting raw predictions."""
    if not results:
        raise ValueError("Cannot write evaluation outputs without model results.")
    experiments = Path(experiments_dir)
    experiments.mkdir(parents=True, exist_ok=True)
    plots = Path(plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    from src.evaluate.plot_results import save_comparison_plots

    plot_paths = save_comparison_plots(results, plots)
    ranked = sorted(
        results,
        key=lambda result: (
            -result.validation.probability_metrics["roc_auc"],
            result.validation.probability_metrics["log_loss"],
        ),
    )
    rank_by_model = {result.model_key: rank for rank, result in enumerate(ranked, start=1)}
    payload = {
        "metadata": dict(metadata or {}),
        "models": [
            {**result.to_dict(), "rank": rank_by_model[result.model_key]}
            for result in ranked
        ],
        "ranking": [
            {
                "rank": rank,
                "model": result.model_key,
                "validation_roc_auc": result.validation.probability_metrics["roc_auc"],
                "validation_log_loss": result.validation.probability_metrics["log_loss"],
            }
            for rank, result in enumerate(ranked, start=1)
        ],
        "best_model": ranked[0].model_key,
        "plot_paths": [str(path) for path in plot_paths],
    }
    json_path = experiments / "model_evaluation_results.json"
    json_tmp = experiments / ".model_evaluation_results.json.tmp"
    with json_tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    json_tmp.replace(json_path)

    metric_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    for result in results:
        for split_result in (result.validation, result.test):
            for metric, value in split_result.probability_metrics.items():
                metric_rows.append(
                    {
                        "model": result.model_key,
                        "rank": rank_by_model[result.model_key],
                        "split": split_result.split,
                        "n_rows": split_result.n_rows,
                        "positive_rate": split_result.positive_rate,
                        "metric": metric,
                        "value": value,
                        "selected_threshold": split_result.selected_threshold,
                    }
                )
            for row in split_result.threshold_metrics:
                threshold_rows.append(
                    {
                        "model": result.model_key,
                        "rank": rank_by_model[result.model_key],
                        "split": split_result.split,
                        "n_rows": split_result.n_rows,
                        "positive_rate": split_result.positive_rate,
                        **row,
                    }
                )

    metrics_path = experiments / "model_metrics.csv"
    metrics_tmp = experiments / ".model_metrics.csv.tmp"
    with metrics_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    metrics_tmp.replace(metrics_path)

    thresholds_path = experiments / "threshold_metrics.csv"
    thresholds_tmp = experiments / ".threshold_metrics.csv.tmp"
    with thresholds_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    thresholds_tmp.replace(thresholds_path)
    return {
        "json": json_path,
        "metrics_csv": metrics_path,
        "thresholds_csv": thresholds_path,
        "plots_dir": plots,
    }


def load_evaluation_results(path: str | Path) -> Dict[str, Any]:
    """Load a previously written JSON result file for notebook resume/reporting."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_evaluation_result_models(path: str | Path) -> List[ModelEvaluationResult]:
    """Load serialized model results for notebook resume/merge."""
    payload = load_evaluation_results(path)
    return [ModelEvaluationResult.from_dict(item) for item in payload.get("models", [])]
