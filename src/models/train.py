"""
Shared Fitting Logic for the Tree-Based CTR Models.

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
import json
import logging
import sys
import time

# Ensure root workspace is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.models.base_model import BaseCTRModel
from src.models.catboost_model import CatBoostCTRModel
from src.models.data_utils import CTRDataset, load_ctr_dataset
from src.models.random_forest_model import RandomForestCTRModel
from src.models.xgboost_model import XGBoostCTRModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """The wrapper class implementing a model, and the config that drives it."""

    model_class: Type[BaseCTRModel]
    config_path: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "catboost": ModelSpec(CatBoostCTRModel, "configs/catboost.yaml"),
    "xgboost": ModelSpec(XGBoostCTRModel, "configs/xgboost.yaml"),
    "random_forest": ModelSpec(RandomForestCTRModel, "configs/random_forest.yaml"),
}


def get_model_class(model_key: str) -> Type[BaseCTRModel]:
    """Return the wrapper class registered for a model key."""
    return MODEL_REGISTRY[_validate_model_key(model_key)].model_class


@dataclass
class FitResult:
    """Everything a single training run produced."""

    model_key: str
    model: BaseCTRModel
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


def _build_kwargs(model_cls, params: Dict[str, Any], random_seed: int) -> Dict[str, Any]:
    """Keep only the config entries the model constructor actually accepts."""
    accepted = set(inspect.signature(model_cls.__init__).parameters) - {
        "self",
        "config",
        "categorical_features",
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

    XGBoost memorizes raw high-cardinality advertiser IDs, so its config drops them in favour of
    the smoothed `*_te` encodings; CatBoost's ordered target statistics regularize them already.
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
    if sample_size is None:
        sample_size = data_cfg.get("sample_size")
    if sample_fraction is None:
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
) -> FitResult:
    """
    Fit the single tree model described by one YAML config, then persist it.

    Args:
        config_path: Path to the model YAML (defaults to the registry path for `model_key`).
        config: Already-parsed config dict, taking precedence over `config_path`.
        model_key: 'catboost' or 'xgboost'. Defaults to the config's own `model:` field.
        processed_dir: Override for the parquet partition directory.
        models_dir: Override for the artifact destination directory.
        use_fe: Override for `data.use_fe`.
        sample_size: Override for `data.sample_size` (0 or None -> full dataset).
        sample_fraction: Override for `data.sample_fraction`.
        random_seed: Override for `data.random_seed`.
        save_artifact: Whether to serialize the fitted model to `models_dir`.
        write_manifest: Whether to write the JSON training manifest.
        dataset: Pre-loaded dataset, to fit several configs without re-reading parquet.

    Returns:
        FitResult: the fitted model, the dataset it was fitted on, and the run manifest.
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

    use_fe = data_cfg.get("use_fe", True) if use_fe is None else use_fe
    seed = data_cfg.get("random_seed", 42) if random_seed is None else random_seed

    logger.info("=" * 70)
    logger.info(f"FIT: {model_cls.__name__}  (config: {config_path or 'in-memory'})")
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

    # Fit-time arguments, not constructor arguments.
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)
    verbose_eval = params.pop("verbose_eval", 50)

    model = model_cls(
        categorical_features=dataset.categorical_features,
        config=params,
        **_build_kwargs(model_cls, params, seed),
    )

    # Forward only the fit-time arguments this wrapper declares: a forest has neither early
    # stopping nor per-round logging, so passing them through would reach sklearn and fail.
    fit_params = inspect.signature(model.fit).parameters
    fit_kwargs: Dict[str, Any] = {}
    if "early_stopping_rounds" in fit_params:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    if "verbose_eval" in fit_params:
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

    # Persist the artifact Task 4 will load.
    basename = paths_cfg.get("model_basename", model_key)
    artifact_path = Path(models_dir or paths_cfg.get("models_dir", "models")) / (
        f"{basename}{'_fe' if use_fe else '_baseline'}.joblib"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if save_artifact:
        try:
            model.save(artifact_path)
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
        "early_stopping_rounds": fit_kwargs.get("early_stopping_rounds"),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "train_seconds": round(elapsed, 2),
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
