"""
CLI Entry Point to Train, Evaluate, and Compare CatBoost and XGBoost CTR Models.

Reads the preprocessed (or feature-engineered) Parquet partitions with Polars,
trains the selected gradient boosting models with validation-based early stopping,
evaluates ROC-AUC / LogLoss / PR-AUC / Brier on val and test, and persists both the
model artifacts and a JSON benchmark summary.

Usage Examples:
    # Quick smoke run on 100k engineered rows, both models
    python -m src.models.run_tree_models --use-fe --sample-size 100000

    # CatBoost only, full engineered dataset
    python -m src.models.run_tree_models --use-fe --sample-size 0 --model catboost

    # XGBoost on a 5% sample with a custom config
    python -m src.models.run_tree_models --use-fe --sample-fraction 0.05 \
        --model xgboost --config configs/tree_models.yaml
"""

from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import inspect
import json
import logging
import sys
import time

# Ensure root workspace is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
import yaml

from src.models.catboost_model import CatBoostCTRModel
from src.models.data_utils import CTRDataset, load_ctr_dataset
from src.models.xgboost_model import XGBoostCTRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_tree_models")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the YAML tree-model configuration, tolerating a missing file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config file not found at {path}, falling back to built-in defaults.")
    return {}


def _build_kwargs(model_cls, cfg: Dict[str, Any], random_seed: int) -> Dict[str, Any]:
    """Keep only the config entries the model constructor actually accepts."""
    accepted = set(inspect.signature(model_cls.__init__).parameters) - {
        "self",
        "config",
        "categorical_features",
    }
    kwargs = {k: v for k, v in cfg.items() if k in accepted}
    ignored = sorted(set(cfg) - accepted)
    if ignored:
        logger.warning(f"Ignoring unsupported {model_cls.__name__} config keys: {ignored}")
    kwargs.setdefault("random_state", random_seed)
    return kwargs


def _scope_dataset(dataset: CTRDataset, drop_features: Optional[list]) -> CTRDataset:
    """
    Return a per-model view of the dataset with `drop_features` removed from every partition.

    XGBoost splits directly on raw categorical levels, so very high-cardinality advertiser
    identifiers let it memorize IDs instead of learning patterns. Dropping them here lets a
    single model keep using the smoothed `*_te` encodings of the same entities, while other
    models (CatBoost, whose ordered target statistics already regularize those columns) keep
    the full feature set.
    """
    drop = [c for c in (drop_features or []) if c in dataset.X_train.columns]
    if not drop:
        return dataset

    logger.info(f"Dropping {len(drop)} high-cardinality feature(s) for this model: {drop}")

    def _drop(df):
        return df.drop(drop) if df is not None else None

    return CTRDataset(
        X_train=_drop(dataset.X_train),
        y_train=dataset.y_train,
        X_val=_drop(dataset.X_val),
        y_val=dataset.y_val,
        X_test=_drop(dataset.X_test),
        y_test=dataset.y_test,
        categorical_features=[c for c in dataset.categorical_features if c not in drop],
        numeric_features=[c for c in dataset.numeric_features if c not in drop],
    )


def _evaluate_splits(model, dataset) -> Dict[str, float]:
    """Evaluate a fitted model on the validation and test partitions."""
    metrics: Dict[str, float] = {}
    if dataset.X_val is not None and dataset.y_val is not None:
        metrics.update(model.evaluate(dataset.X_val, dataset.y_val, dataset_name="Validation"))
    if dataset.X_test is not None and dataset.y_test is not None:
        metrics.update(model.evaluate(dataset.X_test, dataset.y_test, dataset_name="Test"))
    return metrics


def _persist(model, models_path: Path, filename: str) -> None:
    """Serialize a model artifact, logging (but not raising on) I/O failures."""
    save_path = models_path / filename
    try:
        model.save(save_path)
    except Exception as exc:
        logger.warning(f"Could not save model to {save_path}: {exc}")


def run_tree_model_pipeline(
    config: Optional[Dict[str, Any]] = None,
    processed_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    output_metrics_file: Optional[str] = None,
    use_fe: bool = True,
    sample_size: Optional[int] = 100000,
    sample_fraction: Optional[float] = None,
    run_catboost: bool = True,
    run_xgboost: bool = True,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate the CatBoost and XGBoost CTR models.

    Args:
        config: Parsed tree-model configuration dictionary.
        processed_dir: Path to the Parquet partitions (default: from config).
        models_dir: Destination folder for trained model artifacts (default: from config).
        output_metrics_file: File path for the JSON benchmark summary (default: from config).
        use_fe: Whether to train on the engineered partitions (train_fe.parquet, ...).
        sample_size: Optional training row limit (None for the full dataset).
        sample_fraction: Optional sampling fraction applied to every partition.
        run_catboost: Whether to train the CatBoost model.
        run_xgboost: Whether to train the XGBoost model.
        random_seed: Random seed for sampling and model initialization.

    Returns:
        Dict[str, Dict[str, float]]: Comparison table of evaluated metrics per model.
    """
    config = config or {}
    paths_cfg = config.get("paths", {})
    features_cfg = config.get("features", {})
    cat_cfg = dict(config.get("catboost", {}))
    xgb_cfg = dict(config.get("xgboost", {}))

    processed_dir = processed_dir or paths_cfg.get("processed_dir", "data/processed")
    models_path = Path(models_dir or paths_cfg.get("models_dir", "models"))
    models_path.mkdir(parents=True, exist_ok=True)

    out_metrics_path = Path(
        output_metrics_file or paths_cfg.get("results_output", "experiments/tree_models_results.json")
    )
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset in Polars
    logger.info("=" * 65)
    logger.info(f"Loading CTR Dataset splits with Polars (Use Feature Engineering: {use_fe})...")
    logger.info("=" * 65)

    dataset = load_ctr_dataset(
        processed_dir=processed_dir,
        target_col=features_cfg.get("target", "clk"),
        exclude_cols=features_cfg.get("exclude_cols"),
        categorical_cols=features_cfg.get("categorical"),
        numeric_cols=features_cfg.get("numeric"),
        use_fe=use_fe,
        sample_size=sample_size,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
    )

    results: Dict[str, Dict[str, float]] = {}

    # 2. Train CatBoost
    if run_catboost:
        logger.info("\n" + "=" * 65)
        logger.info("TRAINING: CatBoost (ordered target statistics on native categoricals)")
        logger.info("=" * 65)

        early_stopping = cat_cfg.pop("early_stopping_rounds", 100)
        cb_dataset = _scope_dataset(dataset, cat_cfg.pop("drop_features", None))

        cb_model = CatBoostCTRModel(
            categorical_features=cb_dataset.categorical_features,
            config=cat_cfg,
            **_build_kwargs(CatBoostCTRModel, cat_cfg, random_seed),
        )

        start = time.time()
        cb_model.fit(
            X_train=cb_dataset.X_train,
            y_train=cb_dataset.y_train,
            X_val=cb_dataset.X_val,
            y_val=cb_dataset.y_val,
            early_stopping_rounds=early_stopping,
        )
        elapsed = time.time() - start

        cb_metrics = _evaluate_splits(cb_model, cb_dataset)
        cb_metrics["train_seconds"] = round(elapsed, 2)
        cb_metrics["best_iteration"] = int(cb_model.best_iteration_ or 0)
        results["CatBoost"] = cb_metrics

        _persist(cb_model, models_path, "catboost_fe.joblib" if use_fe else "catboost_baseline.joblib")

        try:
            logger.info(
                "Top 15 Feature Importances (CatBoost):\n"
                + str(cb_model.get_feature_importance(top_k=15))
            )
        except Exception as exc:
            logger.debug(f"Could not extract CatBoost feature importances: {exc}")

    # 3. Train XGBoost
    if run_xgboost:
        logger.info("\n" + "=" * 65)
        logger.info("TRAINING: XGBoost (histogram trees with native categorical splits)")
        logger.info("=" * 65)

        early_stopping = xgb_cfg.pop("early_stopping_rounds", 100)
        verbose_eval = xgb_cfg.pop("verbose_eval", 50)
        xgb_dataset = _scope_dataset(dataset, xgb_cfg.pop("drop_features", None))

        xgb_model = XGBoostCTRModel(
            categorical_features=xgb_dataset.categorical_features,
            config=xgb_cfg,
            **_build_kwargs(XGBoostCTRModel, xgb_cfg, random_seed),
        )

        start = time.time()
        xgb_model.fit(
            X_train=xgb_dataset.X_train,
            y_train=xgb_dataset.y_train,
            X_val=xgb_dataset.X_val,
            y_val=xgb_dataset.y_val,
            early_stopping_rounds=early_stopping,
            verbose_eval=verbose_eval,
        )
        elapsed = time.time() - start

        xgb_metrics = _evaluate_splits(xgb_model, xgb_dataset)
        xgb_metrics["train_seconds"] = round(elapsed, 2)
        xgb_metrics["best_iteration"] = int(xgb_model.best_iteration_ or 0)
        results["XGBoost"] = xgb_metrics

        _persist(xgb_model, models_path, "xgboost_fe.joblib" if use_fe else "xgboost_baseline.joblib")

        try:
            logger.info(
                "Top 15 Feature Importances by Gain (XGBoost):\n"
                + str(xgb_model.get_feature_importance(importance_type="gain", top_k=15))
            )
        except Exception as exc:
            logger.debug(f"Could not extract XGBoost feature importances: {exc}")

    # 4. Summary
    logger.info("\n" + "=" * 65)
    logger.info("TREE MODEL BENCHMARK SUMMARY")
    logger.info("=" * 65)

    if results:
        summary_df = pl.DataFrame([{"model": name, **metrics} for name, metrics in results.items()])
        logger.info("\n" + str(summary_df))

    try:
        with open(out_metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved benchmark metrics summary to: {out_metrics_path}")
    except Exception as exc:
        logger.warning(f"Could not save metrics JSON to {out_metrics_path}: {exc}")

    return results


def main() -> None:
    """CLI parser and entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate CatBoost and XGBoost CTR models on Polars parquet partitions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tree_models.yaml",
        help="Path to the tree model YAML configuration.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory containing the processed parquet files (default: from config).",
    )
    parser.add_argument(
        "--use-fe",
        action="store_true",
        default=False,
        help="Train on the engineered datasets (train_fe.parquet, val_fe.parquet, test_fe.parquet).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Destination directory for model artifacts (default: from config).",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="File path for the JSON results summary (default: from config).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "catboost", "xgboost"],
        default="all",
        help="Which model(s) to train (default: 'all').",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Number of training rows to sample (default: 100,000; set 0 for the full dataset).",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional sampling fraction applied to every partition (e.g. 0.05 for 5%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and model initialization (default: 42).",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    sample_size = None if (args.sample_size is not None and args.sample_size <= 0) else args.sample_size

    try:
        run_tree_model_pipeline(
            config=config,
            processed_dir=args.processed_dir,
            models_dir=args.models_dir,
            output_metrics_file=args.output_metrics,
            use_fe=args.use_fe,
            sample_size=sample_size,
            sample_fraction=args.sample_fraction,
            run_catboost=args.model in ("all", "catboost"),
            run_xgboost=args.model in ("all", "xgboost"),
            random_seed=args.seed,
        )
    except Exception as exc:
        logger.error(f"Execution failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
