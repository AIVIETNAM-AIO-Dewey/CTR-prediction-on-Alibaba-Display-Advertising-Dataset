"""
Shared Fitting Logic for CTR Prediction Models.

Turns a model YAML configuration into a fitted, persisted artifact. Supports:
- CatBoost (CatBoostCTRModel)
- XGBoost (XGBoostCTRModel)
- LightGBM (LightGBMModel)
- Logistic Regression (LogisticRegressionModel)

Usage:
    # From CLI:
    python -m src.models.train --model lightgbm --sample-size 10000
    python -m src.models.train --model catboost --sample-size 10000
    python -m src.models.train --model xgboost --sample-size 10000
    python -m src.models.train --model logistic_regression --sample-size 10000

    # In Python / Kaggle Notebook:
    from src.models.train import fit_from_config
    run = fit_from_config("configs/lightgbm.yaml", sample_size=10_000)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import inspect
import json
import logging
import sys
import time

# Ensure root workspace is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import polars as pl
import yaml

from src.models.catboost_model import CatBoostCTRModel
from src.models.lightgbm_model import LightGBMModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.xgboost_model import XGBoostCTRModel

logger = logging.getLogger(__name__)

# Default column exclusions (targets, timestamps, identifiers)
DEFAULT_TARGET_COL = "clk"
DEFAULT_EXCLUDE_COLS = [
    "clk",
    "nonclk",
    "datetime",
    "date",
    "time_stamp",
    "user",
    "userid",
    "user_id",
    "adgroup_id",
    "cms_segid",
]

DEFAULT_CATEGORICAL_COLS = [
    "pid",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
    "cms_group_id",
    "cate_id",
    "brand",
    "customer",
    "campaign_id",
    "is_weekend",
    "day_of_week",
    "hour",
    "gender_x_cate",
    "pid_x_cate",
]

DEFAULT_NUMERIC_COLS = [
    "price",
    "user_adgroup_exposure_seq",
    "user_cate_exposure_seq",
    "price_log",
    "price_ratio_cate",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "cate_id_te",
    "brand_te",
    "customer_te",
    "pid_te",
]


class CTRDataset:
    """Container holding Train, Validation, and Test feature matrices and labels."""

    def __init__(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.categorical_features = list(categorical_features or [])
        self.numeric_features = list(numeric_features or [])

    @property
    def feature_names(self) -> List[str]:
        if hasattr(self.X_train, "columns"):
            return list(self.X_train.columns)
        return []

    def summary(self) -> Dict[str, Any]:
        """Return dataset partition sizes and baseline CTRs."""
        info: Dict[str, Any] = {
            "train_samples": len(self.X_train),
            "num_features": len(self.feature_names),
            "categorical_features": len(self.categorical_features),
            "numeric_features": len(self.numeric_features),
        }
        if hasattr(self.y_train, "mean"):
            info["train_ctr"] = float(self.y_train.mean() * 100)
        if self.X_val is not None:
            info["val_samples"] = len(self.X_val)
            if self.y_val is not None and hasattr(self.y_val, "mean"):
                info["val_ctr"] = float(self.y_val.mean() * 100)
        if self.X_test is not None:
            info["test_samples"] = len(self.X_test)
            if self.y_test is not None and hasattr(self.y_test, "mean"):
                info["test_ctr"] = float(self.y_test.mean() * 100)
        return info


def _resolve_data_dir(processed_dir: Union[str, Path], use_fe: bool = True) -> Path:
    """Find the directory containing parquet partitions locally or on Kaggle."""
    proc_path = Path(processed_dir)
    target_file = "train_fe.parquet" if use_fe else "train.parquet"
    if (proc_path / target_file).exists():
        return proc_path

    # Check relative paths (e.g. from notebooks/)
    rel_path = Path("..") / proc_path
    if (rel_path / target_file).exists():
        return rel_path

    # Check Kaggle input directories
    kaggle_base = Path("/kaggle/input")
    if kaggle_base.exists():
        for candidate in kaggle_base.rglob(target_file):
            if candidate.is_file():
                return candidate.parent

    return proc_path


def load_ctr_dataset(
    processed_dir: Union[str, Path] = "data/processed",
    target_col: str = DEFAULT_TARGET_COL,
    exclude_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    use_fe: bool = True,
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: int = 42,
) -> CTRDataset:
    """
    Load preprocessed Parquet datasets and split into X and y for Train, Val, and Test.
    """
    proc_path = _resolve_data_dir(processed_dir, use_fe=use_fe)
    train_file = "train_fe.parquet" if use_fe else "train.parquet"
    val_file = "val_fe.parquet" if use_fe else "val.parquet"
    test_file = "test_fe.parquet" if use_fe else "test.parquet"

    train_path = proc_path / train_file
    val_path = proc_path / val_file
    test_path = proc_path / test_file

    if not train_path.exists():
        alt_train = proc_path / ("train.parquet" if use_fe else "train_fe.parquet")
        if alt_train.exists():
            train_path = alt_train
            val_path = proc_path / ("val.parquet" if use_fe else "val_fe.parquet")
            test_path = proc_path / ("test.parquet" if use_fe else "test_fe.parquet")
        else:
            raise FileNotFoundError(
                f"Training dataset not found at {train_path}. Run preprocessing first."
            )

    logger.info(f"Loading datasets from {proc_path}...")
    train_pl = pl.read_parquet(train_path)
    val_pl = pl.read_parquet(val_path) if val_path.exists() else None
    test_pl = pl.read_parquet(test_path) if test_path.exists() else None

    # Apply sampling if specified
    if sample_size is not None and 0 < sample_size < len(train_pl):
        logger.info(f"Sampling training set to {sample_size:,} rows...")
        train_pl = train_pl.sample(n=sample_size, seed=random_seed)
        if val_pl is not None:
            val_sample = min(int(sample_size * 0.2), len(val_pl))
            val_pl = val_pl.sample(n=val_sample, seed=random_seed)
        if test_pl is not None:
            test_sample = min(int(sample_size * 0.2), len(test_pl))
            test_pl = test_pl.sample(n=test_sample, seed=random_seed)
    elif sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        logger.info(f"Sampling dataset with fraction {sample_fraction:.2%}...")
        train_pl = train_pl.sample(fraction=sample_fraction, seed=random_seed)
        if val_pl is not None:
            val_pl = val_pl.sample(fraction=sample_fraction, seed=random_seed)
        if test_pl is not None:
            test_pl = test_pl.sample(fraction=sample_fraction, seed=random_seed)

    excluded = set(exclude_cols or DEFAULT_EXCLUDE_COLS)
    all_cols = train_pl.columns
    feature_cols = [c for c in all_cols if c not in excluded and c != target_col]

    cat_candidates = categorical_cols or DEFAULT_CATEGORICAL_COLS
    num_candidates = numeric_cols or DEFAULT_NUMERIC_COLS
    active_cats = [c for c in cat_candidates if c in feature_cols]
    active_nums = [c for c in num_candidates if c in feature_cols]

    # Convert to Pandas for universal model compatibility
    train_pd = train_pl.select(feature_cols + [target_col]).to_pandas()
    X_train = train_pd[feature_cols]
    y_train = train_pd[target_col]

    X_val, y_val = None, None
    if val_pl is not None:
        val_pd = val_pl.select(feature_cols + [target_col]).to_pandas()
        X_val = val_pd[feature_cols]
        y_val = val_pd[target_col]

    X_test, y_test = None, None
    if test_pl is not None:
        test_pd = test_pl.select(feature_cols + [target_col]).to_pandas()
        X_test = test_pd[feature_cols]
        y_test = test_pd[target_col]

    dataset = CTRDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_features=active_cats,
        numeric_features=active_nums,
    )

    summary = dataset.summary()
    logger.info(
        f"CTRDataset Loaded -> Train: {summary['train_samples']:,} rows | "
        f"Val: {summary.get('val_samples', 0):,} rows | Test: {summary.get('test_samples', 0):,} rows | "
        f"Features: {summary['num_features']} ({summary['categorical_features']} cat, {summary['numeric_features']} num)"
    )
    return dataset


@dataclass(frozen=True)
class ModelSpec:
    """The wrapper class implementing a model, and the config that drives it."""

    model_class: Any
    config_path: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "catboost": ModelSpec(CatBoostCTRModel, "configs/catboost.yaml"),
    "xgboost": ModelSpec(XGBoostCTRModel, "configs/xgboost.yaml"),
    "lightgbm": ModelSpec(LightGBMModel, "configs/lightgbm.yaml"),
    "lgb": ModelSpec(LightGBMModel, "configs/lightgbm.yaml"),
    "logistic_regression": ModelSpec(LogisticRegressionModel, "configs/logistic_regression.yaml"),
    "lr": ModelSpec(LogisticRegressionModel, "configs/logistic_regression.yaml"),
}


def get_model_class(model_key: str) -> Any:
    """Return the wrapper class registered for a model key."""
    return MODEL_REGISTRY[_validate_model_key(model_key)].model_class


@dataclass
class FitResult:
    """Everything a single training run produced."""

    model_key: str
    model: Any
    dataset: CTRDataset
    artifact_path: Path
    manifest: Dict[str, Any] = field(default_factory=dict)
    manifest_path: Optional[Path] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a model YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Expected one of: "
            + ", ".join(spec.config_path for spec in MODEL_REGISTRY.values())
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_config_path(model_key: str) -> str:
    """Return the canonical config path for a model key."""
    return MODEL_REGISTRY[_validate_model_key(model_key)].config_path


def _validate_model_key(model_key: str) -> str:
    key = (model_key or "").strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Supported models: {sorted(MODEL_REGISTRY)}"
        )
    return key


def _build_kwargs(model_cls: Any, params: Dict[str, Any], random_seed: int) -> Dict[str, Any]:
    """Keep only the config entries the model constructor actually accepts."""
    accepted = set(inspect.signature(model_cls.__init__).parameters) - {
        "self",
        "config",
        "categorical_features",
        "numeric_features",
    }
    kwargs = {k: v for k, v in params.items() if k in accepted}
    ignored = sorted(set(params) - accepted)
    if ignored:
        logger.warning(
            f"Ignoring unsupported {getattr(model_cls, '__name__', str(model_cls))} config keys: {ignored}"
        )
    kwargs.setdefault("random_state", random_seed)
    return kwargs


def _scope_dataset(dataset: CTRDataset, drop_features: Optional[List[str]]) -> CTRDataset:
    """Return a view of the dataset with `drop_features` removed from every partition."""
    drop = [c for c in (drop_features or []) if c in dataset.X_train.columns]
    if not drop:
        return dataset

    logger.info(f"Dropping {len(drop)} feature(s) for this model: {drop}")

    def _drop(df):
        if df is None:
            return None
        if hasattr(df, "drop"):
            if isinstance(df, pl.DataFrame):
                return df.drop([c for c in drop if c in df.columns])
            return df.drop(columns=[c for c in drop if c in df.columns])
        return df

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


def load_dataset_from_config(
    config: Dict[str, Any],
    processed_dir: Optional[str] = None,
    use_fe: Optional[bool] = None,
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: Optional[int] = None,
    apply_drop_features: bool = True,
) -> CTRDataset:
    """Build the exact dataset view a config describes."""
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    features_cfg = config.get("features", {})

    use_fe = data_cfg.get("use_fe", True) if use_fe is None else use_fe
    seed = data_cfg.get("random_seed", 42) if random_seed is None else random_seed
    if sample_size is None:
        sample_size = data_cfg.get("sample_size")
    if sample_fraction is None:
        sample_fraction = data_cfg.get("sample_fraction")
    if sample_size is not None and sample_size <= 0:
        sample_size = None

    dataset = load_ctr_dataset(
        processed_dir=processed_dir or paths_cfg.get("processed_dir", "data/processed"),
        target_col=features_cfg.get("target", "clk"),
        exclude_cols=features_cfg.get("exclude_cols"),
        categorical_cols=features_cfg.get("categorical"),
        numeric_cols=features_cfg.get("numeric"),
        use_fe=use_fe,
        sample_size=sample_size,
        sample_fraction=sample_fraction,
        random_seed=seed,
    )

    if apply_drop_features:
        dataset = _scope_dataset(dataset, features_cfg.get("drop_features"))
    return dataset


def fit_from_config(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    model_key: Optional[str] = None,
    processed_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    use_fe: Optional[bool] = None,
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: Optional[int] = None,
    save_artifact: bool = True,
    write_manifest: bool = True,
    dataset: Optional[CTRDataset] = None,
    **kwargs: Any,
) -> FitResult:
    """
    Fit any CTR model (CatBoost, XGBoost, LightGBM, Logistic Regression) described by config.

    Args:
        config_path: Path to the model YAML (defaults to registry path for model_key).
        config: Already-parsed config dict, taking precedence over config_path.
        model_key: 'catboost', 'xgboost', 'lightgbm', or 'logistic_regression'.
        processed_dir: Override for the parquet partition directory.
        models_dir: Override for the artifact destination directory.
        use_fe: Override for data.use_fe.
        sample_size: Override for data.sample_size (0 or None -> full dataset).
        sample_fraction: Override for data.sample_fraction.
        random_seed: Override for data.random_seed.
        save_artifact: Whether to serialize the fitted model to models_dir.
        write_manifest: Whether to write the JSON training manifest.
        dataset: Pre-loaded dataset to train without re-reading parquet files.
        **kwargs: Additional hyperparameter overrides passed directly to model config.

    Returns:
        FitResult: the fitted model, dataset, artifact path, and manifest.
    """
    if config is None:
        if config_path is None:
            if model_key is None:
                raise ValueError("Provide one of: config, config_path, or model_key.")
            config_path = default_config_path(model_key)
        config = load_config(config_path)

    model_key = _validate_model_key(model_key or config.get("model", ""))
    model_cls = get_model_class(model_key)

    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    params = dict(config.get("params", {}))

    # Apply any explicit runtime kwargs
    if kwargs:
        params.update(kwargs)

    use_fe = data_cfg.get("use_fe", True) if use_fe is None else use_fe
    seed = data_cfg.get("random_seed", 42) if random_seed is None else random_seed

    logger.info("=" * 70)
    logger.info(f"FIT: {model_cls.__name__}  (model_key: {model_key}, config: {config_path or 'dict'})")
    logger.info("=" * 70)

    if dataset is None:
        dataset = load_dataset_from_config(
            config,
            processed_dir=processed_dir,
            use_fe=use_fe,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
            random_seed=seed,
        )

    early_stopping_rounds = params.pop("early_stopping_rounds", 100)
    verbose_eval = params.pop("verbose_eval", 50)

    init_kwargs = _build_kwargs(model_cls, params, seed)
    sig = inspect.signature(model_cls.__init__).parameters
    if "categorical_features" in sig:
        init_kwargs["categorical_features"] = dataset.categorical_features
    if "numeric_features" in sig:
        init_kwargs["numeric_features"] = dataset.numeric_features
    if "config" in sig:
        init_kwargs["config"] = params

    model = model_cls(**init_kwargs)

    fit_sig = inspect.signature(model.fit).parameters
    fit_kwargs: Dict[str, Any] = {}
    if "early_stopping_rounds" in fit_sig and early_stopping_rounds is not None:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    if "verbose_eval" in fit_sig and verbose_eval is not None:
        fit_kwargs["verbose_eval"] = verbose_eval

    start = time.time()
    model.fit(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        **fit_kwargs,
    )
    elapsed = time.time() - start
    logger.info(f"Training finished in {elapsed:.1f}s.")

    # Evaluate model on validation and test partitions
    metrics: Dict[str, float] = {}
    if hasattr(model, "evaluate"):
        if dataset.X_val is not None and dataset.y_val is not None:
            try:
                val_metrics = model.evaluate(dataset.X_val, dataset.y_val, dataset_name="Validation")
                metrics.update(val_metrics)
                logger.info(f"Validation Metrics: {val_metrics}")
            except Exception as exc:
                logger.warning(f"Could not evaluate on validation set: {exc}")
        if dataset.X_test is not None and dataset.y_test is not None:
            try:
                test_metrics = model.evaluate(dataset.X_test, dataset.y_test, dataset_name="Test")
                metrics.update(test_metrics)
                logger.info(f"Test Metrics: {test_metrics}")
            except Exception as exc:
                logger.warning(f"Could not evaluate on test set: {exc}")

    # Persist artifact
    basename = paths_cfg.get("model_basename", model_key)
    artifact_path = Path(models_dir or paths_cfg.get("models_dir", "models")) / (
        f"{basename}{'_fe' if use_fe else '_baseline'}.joblib"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if save_artifact:
        try:
            model.save(artifact_path)
            logger.info(f"Model successfully saved to {artifact_path}")
        except Exception as exc:
            logger.warning(f"Could not save model to {artifact_path}: {exc}")

    manifest = {
        "model": model_key,
        "config_path": str(config_path) if config_path else None,
        "artifact_path": str(artifact_path),
        "use_fe": bool(use_fe),
        "random_seed": seed,
        "train_rows": int(len(dataset.X_train)),
        "val_rows": int(len(dataset.X_val)) if dataset.X_val is not None else 0,
        "test_rows": int(len(dataset.X_test)) if dataset.X_test is not None else 0,
        "n_features": len(dataset.feature_names),
        "feature_names": list(dataset.feature_names),
        "categorical_features": list(dataset.categorical_features),
        "numeric_features": list(dataset.numeric_features),
        "dropped_features": list(config.get("features", {}).get("drop_features") or []),
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "train_seconds": round(elapsed, 2),
        "metrics": metrics,
        "params": params,
    }

    manifest_path = None
    if write_manifest:
        manifest_path = Path(
            paths_cfg.get("manifest_output", f"experiments/{model_key}_run.json")
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved training manifest to: {manifest_path}")
        except Exception as exc:
            logger.warning(f"Could not save manifest to {manifest_path}: {exc}")
            manifest_path = None

    # Update summary benchmark reports in experiments/
    display_name = {
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "logistic_regression": "LogisticRegression",
    }.get(model_key, model_key)

    report_entry = {
        **metrics,
        "train_seconds": round(elapsed, 2),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    }

    for report_file in ["experiments/tree_models_results.json", "experiments/models_results.json"]:
        r_path = Path(report_file)
        if r_path.exists() or report_file == "experiments/models_results.json":
            r_data = {}
            if r_path.exists():
                try:
                    with open(r_path, "r", encoding="utf-8") as f:
                        r_data = json.load(f)
                except Exception:
                    r_data = {}
            r_data[display_name] = report_entry
            try:
                r_path.parent.mkdir(parents=True, exist_ok=True)
                with open(r_path, "w", encoding="utf-8") as f:
                    json.dump(r_data, f, indent=2)
                logger.info(f"Updated benchmark report in {r_path}")
            except Exception as exc:
                logger.debug(f"Could not update report at {r_path}: {exc}")

    logger.info(
        f"Done. Artifact: {artifact_path} | best_iteration={manifest['best_iteration']} | "
        f"{manifest['train_rows']:,} train rows | {manifest['n_features']} features."
    )

    return FitResult(
        model_key=model_key,
        model=model,
        dataset=dataset,
        artifact_path=artifact_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def build_arg_parser(model_key: Optional[str] = None, description: Optional[str] = None):
    """Shared CLI surface for all model training runs."""
    import argparse

    default_key = model_key or "catboost"
    desc = description or f"Fit CTR model ({default_key})."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--model",
        type=str,
        default=model_key,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture to train (catboost, xgboost, lightgbm, logistic_regression).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path(default_key) if model_key else None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Directory holding parquet partitions.")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Destination directory for model artifact.")
    fe = parser.add_mutually_exclusive_group()
    fe.add_argument("--use-fe", dest="use_fe", action="store_true", default=None,
                    help="Train on engineered partitions (train_fe.parquet, ...).")
    fe.add_argument("--no-fe", dest="use_fe", action="store_false",
                    help="Train on plain preprocessed partitions.")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Training rows to sample; 0 for full dataset.")
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help="Sampling fraction applied to every partition, e.g. 0.05.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling and model init.")
    parser.add_argument("--no-save", action="store_true",
                        help="Fit without writing the model artifact to disk.")
    return parser


def run_cli(model_key: Optional[str] = None, description: Optional[str] = None) -> None:
    """Parse CLI arguments and fit one model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = build_arg_parser(model_key, description).parse_args()
    chosen_model = args.model or model_key or "catboost"

    try:
        fit_from_config(
            config_path=args.config,
            model_key=chosen_model,
            processed_dir=args.processed_dir,
            models_dir=args.models_dir,
            use_fe=args.use_fe,
            sample_size=args.sample_size,
            sample_fraction=args.sample_fraction,
            random_seed=args.seed,
            save_artifact=not args.no_save,
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
