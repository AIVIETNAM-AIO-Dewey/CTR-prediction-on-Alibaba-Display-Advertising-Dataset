"""
Unified CLI Entry Point to Train, Evaluate, and Compare Baseline Models.
Runs Logistic Regression and LightGBM models on processed or feature-engineered CTR datasets using Polars.
"""

from pathlib import Path
from typing import Dict, Optional
import argparse
import json
import logging
import sys
import polars as pl

from src.models.data_utils import load_ctr_dataset
from src.models.lightgbm_model import LightGBMCTRModel
from src.models.logistic_regression_model import LogisticRegressionCTRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_baseline_pipeline(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    output_metrics_file: str = "experiments/baseline_results.json",
    use_fe: bool = False,
    sample_size: Optional[int] = 100000,
    sample_fraction: Optional[float] = None,
    run_lr: bool = True,
    run_lgb: bool = True,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Execute training and evaluation of baseline models with native Polars DataFrames.

    Args:
        processed_dir: Path to preprocessed Parquet directory.
        models_dir: Destination folder for trained model artifacts.
        output_metrics_file: File path to save evaluation summary.
        use_fe: Whether to use engineered feature datasets (train_fe.parquet).
        sample_size: Optional row limit for fast execution.
        sample_fraction: Optional fraction of full dataset.
        run_lr: Whether to train Logistic Regression baseline.
        run_lgb: Whether to train LightGBM baseline.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict[str, Dict[str, float]]: Comparison table of evaluated metrics.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    out_metrics_path = Path(output_metrics_file)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset in Polars
    logger.info("=" * 65)
    logger.info(
        f"Loading CTR Dataset splits with Polars (Use Feature Engineering: {use_fe})..."
    )
    logger.info("=" * 65)

    dataset = load_ctr_dataset(
        processed_dir=processed_dir,
        use_fe=use_fe,
        sample_size=sample_size,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
    )

    results: Dict[str, Dict[str, float]] = {}

    # 2. Train Logistic Regression Baseline
    if run_lr:
        logger.info("\n" + "=" * 65)
        logger.info("TRAINING: Logistic Regression Baseline (Polars)")
        logger.info("=" * 65)

        lr_model = LogisticRegressionCTRModel(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            categorical_features=dataset.categorical_features,
            numeric_features=dataset.numeric_features,
            random_state=random_seed,
        )

        lr_model.fit(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
        )

        lr_metrics = {}
        if dataset.X_val is not None and dataset.y_val is not None:
            lr_metrics.update(
                lr_model.evaluate(dataset.X_val, dataset.y_val, dataset_name="Validation")
            )
        if dataset.X_test is not None and dataset.y_test is not None:
            lr_metrics.update(
                lr_model.evaluate(dataset.X_test, dataset.y_test, dataset_name="Test")
            )

        lr_save_path = models_path / (
            "logistic_regression_fe.joblib" if use_fe else "logistic_regression_baseline.joblib"
        )
        try:
            lr_model.save(lr_save_path)
        except Exception as e:
            logger.warning(f"Could not save model to {lr_save_path}: {e}")

        results["LogisticRegression"] = lr_metrics

        # Log Top influential coefficients (Polars DataFrame)
        try:
            top_coefs = lr_model.get_top_coefficients(top_k=10)
            logger.info(
                f"Top 10 Influential Features (Logistic Regression):\n{top_coefs}"
            )
        except Exception as e:
            logger.debug(f"Could not extract coefficients: {e}")

    # 3. Train LightGBM Baseline
    if run_lgb:
        logger.info("\n" + "=" * 65)
        logger.info("TRAINING: LightGBM Baseline (Polars)")
        logger.info("=" * 65)

        lgb_model = LightGBMCTRModel(
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            categorical_features=dataset.categorical_features,
            random_state=random_seed,
        )

        lgb_model.fit(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        lgb_metrics = {}
        if dataset.X_val is not None and dataset.y_val is not None:
            lgb_metrics.update(
                lgb_model.evaluate(dataset.X_val, dataset.y_val, dataset_name="Validation")
            )
        if dataset.X_test is not None and dataset.y_test is not None:
            lgb_metrics.update(
                lgb_model.evaluate(dataset.X_test, dataset.y_test, dataset_name="Test")
            )

        lgb_save_path = models_path / (
            "lightgbm_fe.joblib" if use_fe else "lightgbm_baseline.joblib"
        )
        try:
            lgb_model.save(lgb_save_path)
        except Exception as e:
            logger.warning(f"Could not save model to {lgb_save_path}: {e}")

        results["LightGBM"] = lgb_metrics

        # Log Top feature importances (Polars DataFrame)
        try:
            top_imp = lgb_model.get_feature_importance(importance_type="gain", top_k=10)
            logger.info(
                f"Top 10 Feature Importances by Gain (LightGBM):\n{top_imp}"
            )
        except Exception as e:
            logger.debug(f"Could not extract feature importances: {e}")

    # 4. Save and Display Results Summary using Polars
    logger.info("\n" + "=" * 65)
    logger.info("BASELINE BENCHMARK SUMMARY (Polars)")
    logger.info("=" * 65)

    if results:
        summary_rows = []
        for model_name, metrics in results.items():
            row = {"model": model_name}
            row.update(metrics)
            summary_rows.append(row)
        summary_df = pl.DataFrame(summary_rows)
        logger.info("\n" + str(summary_df))

    try:
        with open(out_metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved benchmark metrics summary to: {out_metrics_path}")
    except Exception as e:
        logger.warning(f"Could not save metrics JSON to {out_metrics_path}: {e}")

    return results


def main() -> None:
    """CLI parser and entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Logistic Regression and LightGBM baselines for CTR Prediction using Polars."
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to directory containing processed parquet files.",
    )
    parser.add_argument(
        "--use-fe",
        action="store_true",
        default=False,
        help="Train on feature-engineered datasets (train_fe.parquet, val_fe.parquet, test_fe.parquet).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Destination directory to save model artifacts.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="experiments/baseline_results.json",
        help="File path to save JSON results summary.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "lr", "lightgbm"],
        default="all",
        help="Model baseline to execute (default: 'all').",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Number of training rows to sample (default: 100,000; set 0 for full dataset).",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional sampling fraction (e.g. 0.05 for 5%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and model initialization (default: 42).",
    )

    args = parser.parse_args()

    sample_size = (
        None if (args.sample_size is not None and args.sample_size <= 0) else args.sample_size
    )
    run_lr = args.model in ("all", "lr")
    run_lgb = args.model in ("all", "lightgbm")

    try:
        run_baseline_pipeline(
            processed_dir=args.processed_dir,
            models_dir=args.models_dir,
            output_metrics_file=args.output_metrics,
            use_fe=args.use_fe,
            sample_size=sample_size,
            sample_fraction=args.sample_fraction,
            run_lr=run_lr,
            run_lgb=run_lgb,
            random_seed=args.seed,
        )
    except Exception as exc:
        logger.error(f"Execution failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
