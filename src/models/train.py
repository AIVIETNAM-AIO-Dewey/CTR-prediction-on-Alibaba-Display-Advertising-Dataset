"""
Shared fitting logic for CTR models.

Turns one model YAML configuration into a fitted, persisted artifact (Task 3). Computes no
metrics: evaluation is Task 4 (src/evaluate/) and tuning is Task 5 (experiments/tune_optuna.py).
One config == one model, so a run never starts another model as a side effect.

Backs the CLI entry points (run_catboost, run_xgboost, run_random_forest) and notebooks:

    from src.models.train import fit_from_config
    run = fit_from_config("configs/catboost.yaml", sample_size=200_000)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import inspect
import importlib
import json
import logging
import sys
import time

# Ensure root workspace is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.models.data_utils import (
    CTRDataset,
    ctr_partition_rows,
    inspect_ctr_features,
    load_ctr_dataset,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """The wrapper class implementing a model, and the config that drives it."""

    module: str
    class_name: str
    config_path: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "catboost": ModelSpec(
        "src.models.catboost_model", "CatBoostCTRModel", "configs/catboost.yaml"
    ),
    "xgboost": ModelSpec(
        "src.models.xgboost_model", "XGBoostCTRModel", "configs/xgboost.yaml"
    ),
    "random_forest": ModelSpec(
        "src.models.random_forest_model",
        "RandomForestCTRModel",
        "configs/random_forest.yaml",
    ),
    "lightgbm": ModelSpec(
        "src.models.lightgbm_model", "LightGBMModel", "configs/lightgbm.yaml"
    ),
    "logistic_regression": ModelSpec(
        "src.models.logistic_regression_model",
        "LogisticRegressionModel",
        "configs/logistic_regression.yaml",
    ),
}


def get_model_class(model_key: str) -> Type[Any]:
    """Return the wrapper class registered for a model key."""
    spec = MODEL_REGISTRY[_validate_model_key(model_key)]
    return getattr(importlib.import_module(spec.module), spec.class_name)


@dataclass
class FitResult:
    """Everything a single training run produced."""

    model_key: str
    model: Any
    dataset: Optional[CTRDataset]
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


def _build_kwargs(model_cls, params: Dict[str, Any], random_seed: int) -> Dict[str, Any]:
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
        logger.warning(f"Ignoring unsupported {model_cls.__name__} config keys: {ignored}")
    kwargs.setdefault("random_state", random_seed)
    return kwargs


def _scope_dataset(dataset: CTRDataset, drop_features: Optional[List[str]]) -> CTRDataset:
    """
    Return a view of the dataset with `drop_features` removed from every partition.

    Each config can remove raw high-cardinality IDs before fitting its model.
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


def load_dataset_from_config(
    config: Dict[str, Any],
    processed_dir: Optional[str] = None,
    use_fe: Optional[bool] = None,
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: Optional[int] = None,
    apply_drop_features: bool = True,
    splits: Optional[List[str]] = None,
) -> CTRDataset:
    """
    Build the exact dataset view a config describes.

    Exposed separately so Task 4 can rebuild a model's feature scope without retraining it.
    """
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    features_cfg = config.get("features", {})

    use_fe = data_cfg.get("use_fe", True) if use_fe is None else use_fe
    seed = data_cfg.get("random_seed", 42) if random_seed is None else random_seed
    # Explicit CLI/API overrides win over the alternate sampling mode from YAML.
    if sample_fraction is not None:
        sample_size = None
    elif sample_size is not None:
        sample_fraction = None
    else:
        sample_size = data_cfg.get("sample_size")
        sample_fraction = data_cfg.get("sample_fraction")
    # 0 / negative means "use the full dataset"
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
        splits=splits,
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
    run_signature: Optional[str] = None,
    training_signature: Optional[str] = None,
    save_artifact: bool = True,
    write_manifest: bool = True,
    dataset: Optional[CTRDataset] = None,
    splits: Optional[List[str]] = None,
) -> FitResult:
    """
    Fit the single tree model described by one YAML config, then persist it.

    Args:
        config_path: Path to the model YAML (defaults to the registry path for `model_key`).
        config: Already-parsed config dict, taking precedence over `config_path`.
        model_key: Registry key for one model. Defaults to the config's `model:` field.
        processed_dir: Override for the parquet partition directory.
        models_dir: Override for the artifact destination directory.
        use_fe: Override for `data.use_fe`.
        sample_size: Override for `data.sample_size` (0 or None -> full dataset).
        sample_fraction: Override for `data.sample_fraction`.
        random_seed: Override for `data.random_seed`.
        run_signature: Deprecated alias for ``training_signature``.
        training_signature: Identity used to validate resumable artifacts.
        save_artifact: Whether to serialize the fitted model to `models_dir`.
        write_manifest: Whether to write the JSON training manifest.
        dataset: Pre-loaded dataset, to fit several configs without re-reading parquet.
        splits: Optional partitions to load when ``dataset`` is not supplied. The
            default keeps the historical train/validation/test behaviour.

    Returns:
        FitResult: the fitted model, the dataset it was fitted on, and the run manifest.
    """
    if config is None:
        if config_path is None:
            if model_key is None:
                raise ValueError("Provide one of: config, config_path, or model_key.")
            config_path = default_config_path(model_key)
        config = load_config(config_path)

    if training_signature is None:
        training_signature = run_signature

    model_key = _validate_model_key(model_key or config.get("model", ""))
    model_cls = get_model_class(model_key)

    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    params = dict(config.get("params", {}))

    use_fe = data_cfg.get("use_fe", True) if use_fe is None else use_fe
    seed = data_cfg.get("random_seed", 42) if random_seed is None else random_seed
    effective_requested_sample = data_cfg.get("sample_size") if sample_size is None else sample_size
    effective_requested_fraction = data_cfg.get("sample_fraction") if sample_fraction is None else sample_fraction
    if model_key == "random_forest" and (effective_requested_sample is None or effective_requested_sample <= 0) and effective_requested_fraction is None:
        # sklearn forests are not incremental; avoid multiplying their dense matrix through
        # joblib workers in the full-data path. This changes parallelism only, never row scope.
        params["n_jobs"] = 1
        logger.info("Random Forest full-data guard: forcing n_jobs=1; no sampling fallback is allowed.")

    logger.info("=" * 70)
    logger.info(f"FIT: {model_cls.__name__}  (config: {config_path or 'in-memory'})")
    logger.info("=" * 70)

    streaming_lr = model_key == "logistic_regression" and bool(params.get("streaming", False))
    stream_features: List[str] = []
    stream_cats: List[str] = []
    stream_nums: List[str] = []
    if streaming_lr:
        configured_sample = data_cfg.get("sample_size") if sample_size is None else sample_size
        configured_fraction = data_cfg.get("sample_fraction") if sample_fraction is None else sample_fraction
        if (configured_sample is not None and configured_sample > 0) or configured_fraction is not None:
            raise ValueError("Streaming logistic regression requires full data; sampling is disabled.")
        feature_cfg = config.get("features", {})
        stream_excluded = list(feature_cfg.get("exclude_cols") or []) + list(
            feature_cfg.get("drop_features") or []
        )
        stream_features, stream_cats, stream_nums = inspect_ctr_features(
            processed_dir or paths_cfg.get("processed_dir", "data/processed"),
            target_col=feature_cfg.get("target", "clk"),
            exclude_cols=stream_excluded,
            categorical_cols=feature_cfg.get("categorical"),
            numeric_cols=feature_cfg.get("numeric"),
            use_fe=use_fe,
        )
        dataset = None
    elif dataset is None:
        dataset = load_dataset_from_config(
            config,
            processed_dir=processed_dir,
            use_fe=use_fe,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
            random_seed=seed,
            splits=splits,
        )
    else:
        dataset = _scope_dataset(
            dataset, config.get("features", {}).get("drop_features")
        )

    # Fit-time arguments, not constructor arguments.
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)
    verbose_eval = params.pop("verbose_eval", 50)

    constructor_params = inspect.signature(model_cls.__init__).parameters
    model_kwargs = _build_kwargs(model_cls, params, seed)
    if "categorical_features" in constructor_params:
        model_kwargs["categorical_features"] = stream_cats if streaming_lr else dataset.categorical_features
    if "numeric_features" in constructor_params:
        model_kwargs["numeric_features"] = stream_nums if streaming_lr else dataset.numeric_features
    if "config" in constructor_params:
        model_kwargs["config"] = config
    model = model_cls(**model_kwargs)

    # Forward only the fit-time arguments this wrapper declares: a forest has neither early
    # stopping nor per-round logging, so passing them through would reach sklearn and fail.
    fit_params = inspect.signature(model.fit).parameters
    fit_kwargs: Dict[str, Any] = {}
    if "early_stopping_rounds" in fit_params:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    if "verbose_eval" in fit_params:
        fit_kwargs["verbose_eval"] = verbose_eval

    start = time.time()
    if streaming_lr:
        model.fit_streaming(
            processed_dir or paths_cfg.get("processed_dir", "data/processed"),
            target_col=config.get("features", {}).get("target", "clk"),
            use_fe=use_fe,
            batch_size=params.get("stream_batch_size"),
            epochs=params.get("stream_epochs"),
        )
    else:
        model.fit(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            **fit_kwargs,
        )
    elapsed = time.time() - start
    logger.info(f"Training finished in {elapsed:.1f}s.")

    if streaming_lr:
        effective_sample_size = None
        effective_sample_fraction = None
    elif sample_fraction is not None:
        effective_sample_size = None
        effective_sample_fraction = sample_fraction
    elif sample_size is not None:
        effective_sample_size = sample_size
        effective_sample_fraction = None
    else:
        effective_sample_size = data_cfg.get("sample_size")
        effective_sample_fraction = data_cfg.get("sample_fraction")
    if effective_sample_size is not None and effective_sample_size <= 0:
        effective_sample_size = None

    # Persist the artifact Task 4 will load.
    basename = paths_cfg.get("model_basename", model_key)
    artifact_path = Path(models_dir or paths_cfg.get("models_dir", "models")) / (
        f"{basename}{'_fe' if use_fe else '_baseline'}.joblib"
    )
    if save_artifact:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_artifact = artifact_path.with_name(f".{artifact_path.name}.tmp")
        model.save(temporary_artifact)
        temporary_artifact.replace(artifact_path)

    manifest = {
        "model": model_key,
        "config_path": str(config_path) if config_path else None,
        "artifact_path": str(artifact_path),
        "use_fe": bool(use_fe),
        "random_seed": seed,
        "training_signature": training_signature,
        # Keep the old key in manifests written by callers that still inspect
        # it, while the new checkpoint validator requires training_signature.
        "run_signature": training_signature,
        "train_rows": (
            ctr_partition_rows(processed_dir or paths_cfg.get("processed_dir", "data/processed"), "train", use_fe=use_fe)
            if streaming_lr else int(len(dataset.X_train))
        ),
        "val_rows": (
            ctr_partition_rows(processed_dir or paths_cfg.get("processed_dir", "data/processed"), "val", use_fe=use_fe)
            if streaming_lr and splits and "val" in splits
            else (int(len(dataset.X_val)) if dataset is not None and dataset.X_val is not None else 0)
        ),
        "test_rows": int(len(dataset.X_test)) if dataset is not None and dataset.X_test is not None else 0,
        "n_features": len(stream_features) if streaming_lr else len(dataset.feature_names),
        "feature_names": list(stream_features) if streaming_lr else list(dataset.feature_names),
        "categorical_features": list(stream_cats) if streaming_lr else list(dataset.categorical_features),
        "numeric_features": list(stream_nums) if streaming_lr else list(dataset.numeric_features),
        "fit_mode": "streaming_sgd" if streaming_lr else "in_memory",
        "stream_batch_size": params.get("stream_batch_size") if streaming_lr else None,
        "stream_epochs": params.get("stream_epochs") if streaming_lr else None,
        "stream_imputation_method": getattr(model, "stream_imputation_method_", None),
        "dropped_features": list(config.get("features", {}).get("drop_features") or []),
        "early_stopping_rounds": fit_kwargs.get("early_stopping_rounds"),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "train_seconds": round(elapsed, 2),
        "params": params,
        "sampling": {
            "sample_size": effective_sample_size,
            "sample_fraction": effective_sample_fraction,
            "random_seed": seed,
        },
        "gpu_overrides": {
            key: params[key]
            for key in ("device", "task_type", "devices", "rsm")
            if key in params
        },
    }

    manifest_path = None
    if write_manifest:
        manifest_path = Path(
            paths_cfg.get("manifest_output", f"experiments/{model_key}_run.json")
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
            with temporary_manifest.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            temporary_manifest.replace(manifest_path)
            logger.info(f"Saved training manifest to: {manifest_path}")
        except Exception as exc:
            logger.warning(f"Could not save manifest to {manifest_path}: {exc}")
            manifest_path = None

    logger.info(
        f"Done. Artifact: {artifact_path} | best_iteration={manifest['best_iteration']} | "
        f"{manifest['train_rows']:,} train rows | {manifest['n_features']} features. "
        "Metrics are Task 4 (src/evaluate/)."
    )

    return FitResult(
        model_key=model_key,
        model=model,
        dataset=dataset,
        artifact_path=artifact_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def build_arg_parser(model_key: str, description: str):
    """Shared CLI surface for the per-model entry points."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path(model_key),
        help=f"Path to the {model_key} YAML configuration.",
    )
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Directory holding the parquet partitions (default: from config).")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Destination directory for the model artifact (default: from config).")
    fe = parser.add_mutually_exclusive_group()
    fe.add_argument("--use-fe", dest="use_fe", action="store_true", default=None,
                    help="Train on the engineered partitions (train_fe.parquet, ...).")
    fe.add_argument("--no-fe", dest="use_fe", action="store_false",
                    help="Train on the plain preprocessed partitions.")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Training rows to sample; 0 for the full dataset (default: from config).")
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help="Sampling fraction applied to every partition, e.g. 0.05 (default: from config).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling and model init (default: from config).")
    parser.add_argument("--no-save", action="store_true",
                        help="Fit without writing the model artifact to disk.")
    return parser


def run_cli(model_key: str, description: str) -> None:
    """Parse CLI arguments and fit exactly one model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = build_arg_parser(model_key, description).parse_args()

    try:
        fit_from_config(
            config_path=args.config,
            model_key=model_key,
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
