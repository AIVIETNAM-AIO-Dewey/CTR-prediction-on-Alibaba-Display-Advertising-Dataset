"""LightGBM wrapper used by the CTR training pipeline."""

from __future__ import annotations

import logging
import inspect
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import yaml

logger = logging.getLogger(__name__)

Frame = Union[pl.DataFrame, pd.DataFrame, np.ndarray]
Labels = Union[pl.Series, pd.Series, np.ndarray]


class LightGBMModel:
    """Binary LightGBM model with stable mixed-type preprocessing."""

    def __init__(
        self,
        objective: str = "binary",
        metric: Union[str, List[str]] = ("binary_logloss", "auc"),
        boosting_type: str = "gbdt",
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_estimators: int = 1000,
        scale_pos_weight: float = 1.0,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = -1,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50,
        config: Optional[Dict[str, Any]] = None,
        **extra_kwargs: Any,
    ) -> None:
        self.model_name = "LightGBM"
        self.objective = objective
        self.metric = list(metric) if isinstance(metric, (list, tuple)) else [metric]
        self.boosting_type = boosting_type
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.categorical_features = list(categorical_features or [])
        self.numeric_features = list(numeric_features or [])
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.config = config or {}
        self.extra_kwargs = dict(extra_kwargs)

        self.feature_names: List[str] = []
        self.active_categorical_features_: List[str] = []
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.estimator: Optional[lgb.LGBMClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.evals_result_: Dict[str, Any] = {}
        self.is_fitted = False

    @staticmethod
    def _read_config(config_path_or_dict: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config_path_or_dict, (str, Path)):
            with open(config_path_or_dict, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        return dict(config_path_or_dict)

    @classmethod
    def load_dataset(
        cls,
        config_path_or_dict: Union[str, Path, Dict[str, Any]],
        sample_size: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        data_dir: Optional[Union[str, Path]] = None,
        use_fe: Optional[bool] = None,
    ) -> Tuple[pl.DataFrame, pl.Series, Optional[pl.DataFrame], Optional[pl.Series]]:
        from src.models.train import load_dataset_from_config

        dataset = load_dataset_from_config(
            config=cls._read_config(config_path_or_dict),
            processed_dir=str(data_dir) if data_dir is not None else None,
            use_fe=use_fe,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
        )
        return dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val

    @classmethod
    def from_config(
        cls,
        config_path_or_dict: Union[str, Path, Dict[str, Any]],
        **kwargs: Any,
    ) -> "LightGBMModel":
        cfg = cls._read_config(config_path_or_dict)
        params = dict(cfg.get("params", {}))
        features = cfg.get("features", {})
        dropped = set(features.get("drop_features") or [])
        params.setdefault("random_state", cfg.get("data", {}).get("random_seed", 42))
        params.update(kwargs)
        return cls(
            categorical_features=[c for c in features.get("categorical", []) if c not in dropped],
            numeric_features=[c for c in features.get("numeric", []) if c not in dropped],
            config=cfg,
            **params,
        )

    @classmethod
    def fit_from_config(
        cls,
        config_path_or_dict: Union[str, Path, Dict[str, Any]] = "configs/lightgbm.yaml",
        sample_size: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        data_dir: Optional[Union[str, Path]] = None,
        save_artifact: bool = True,
        models_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Tuple["LightGBMModel", Dict[str, float]]:
        """Load, fit and optionally persist one model; evaluation remains separate."""
        cfg = cls._read_config(config_path_or_dict)
        use_fe = kwargs.pop("use_fe", None)
        X_train, y_train, X_val, y_val = cls.load_dataset(
            cfg,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
            data_dir=data_dir,
            use_fe=use_fe,
        )
        model = cls.from_config(cfg, **kwargs)
        start = time.time()
        model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        logger.info("[%s] Training finished in %.2fs.", cls.__name__, time.time() - start)

        if save_artifact:
            paths = cfg.get("paths", {})
            out_dir = Path(models_dir or paths.get("models_dir", "models"))
            effective_use_fe = cfg.get("data", {}).get("use_fe", True) if use_fe is None else use_fe
            basename = paths.get("model_basename", "lightgbm")
            model.save(out_dir / f"{basename}{'_fe' if effective_use_fe else '_baseline'}.joblib")
        return model, {}

    def _to_frame(self, data: Frame, *, fitting: bool) -> pl.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = pl.from_pandas(data)
        elif isinstance(data, pl.DataFrame):
            frame = data
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("Expected a two-dimensional numpy array.")
            names = (
                [f"f_{index}" for index in range(data.shape[1])]
                if fitting
                else self.feature_names
            )
            if len(names) != data.shape[1]:
                raise ValueError(f"Expected {len(names)} features, received {data.shape[1]}.")
            frame = pl.DataFrame(data, schema=names, orient="row")
        else:
            raise TypeError("Expected a Polars DataFrame, pandas DataFrame, or numpy ndarray.")

        if fitting:
            if not frame.columns:
                raise ValueError("Training data must contain at least one feature.")
            self.feature_names = list(frame.columns)
        else:
            missing = [name for name in self.feature_names if name not in frame.columns]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            frame = frame.select(self.feature_names)
        return frame

    def _prepare_dataframe(self, data: Frame, *, fitting: bool = False) -> pl.DataFrame:
        frame = self._to_frame(data, fitting=fitting)
        if fitting:
            configured = [c for c in self.categorical_features if c in frame.columns]
            detected = [
                c
                for c, dtype in frame.schema.items()
                if dtype in (pl.String, pl.Categorical, pl.Enum, pl.Object)
            ]
            self.active_categorical_features_ = list(dict.fromkeys(configured + detected))
            self.category_maps_ = {}
            for column in self.active_categorical_features_:
                values = (
                    frame.get_column(column)
                    .cast(pl.String, strict=False)
                    .fill_null("__MISSING__")
                    .unique()
                    .sort()
                    .to_list()
                )
                self.category_maps_[column] = {
                    value: index for index, value in enumerate(values)
                }

        active_cats = set(self.active_categorical_features_)
        expressions: List[pl.Expr] = []
        for column, dtype in frame.schema.items():
            if column in active_cats:
                expressions.append(
                    pl.col(column)
                    .cast(pl.String, strict=False)
                    .fill_null("__MISSING__")
                    .replace_strict(
                        self.category_maps_[column],
                        default=-1,
                        return_dtype=pl.Int32,
                    )
                    .alias(column)
                )
            elif dtype == pl.Boolean:
                expressions.append(pl.col(column).cast(pl.Int8).alias(column))
        return frame.with_columns(expressions) if expressions else frame

    @staticmethod
    def _labels(values: Labels) -> np.ndarray:
        if isinstance(values, (pl.Series, pd.Series)):
            return values.to_numpy().ravel()
        return np.asarray(values).ravel()

    def fit(
        self,
        X: Optional[Frame] = None,
        y: Optional[Labels] = None,
        X_train: Optional[Frame] = None,
        y_train: Optional[Labels] = None,
        X_val: Optional[Frame] = None,
        y_val: Optional[Labels] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: Optional[int] = None,
        **kwargs: Any,
    ) -> "LightGBMModel":
        X = X_train if X is None else X
        y = y_train if y is None else y
        if X is None or y is None:
            raise ValueError("Training features and labels must be provided.")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")

        X_tr = self._prepare_dataframe(X, fitting=True)
        y_tr = self._labels(y)
        if len(X_tr) != len(y_tr):
            raise ValueError("Training features and labels have different row counts.")

        early_stopping_rounds = (
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds
        )
        verbose_eval = self.verbose_eval if verbose_eval is None else verbose_eval
        callbacks: List[Any] = []
        if verbose_eval and verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))

        eval_set = None
        if X_val is not None and y_val is not None:
            X_va = self._prepare_dataframe(X_val)
            y_va = self._labels(y_val)
            if len(X_va) != len(y_va):
                raise ValueError("Validation features and labels have different row counts.")
            eval_set = [(X_va, y_va)]
            if early_stopping_rounds and early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        first_metric_only=False,
                        verbose=bool(verbose_eval and verbose_eval > 0),
                    )
                )

        params: Dict[str, Any] = {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_estimators": self.n_estimators,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            **self.extra_kwargs,
        }
        self.estimator = lgb.LGBMClassifier(**params)
        fit_kwargs: Dict[str, Any] = {
            "callbacks": callbacks,
            "categorical_feature": self.active_categorical_features_ or "auto",
            **kwargs,
        }
        if eval_set is not None:
            if "eval_X" in inspect.signature(self.estimator.fit).parameters:
                fit_kwargs["eval_X"] = eval_set[0][0]
                fit_kwargs["eval_y"] = eval_set[0][1]
            else:
                fit_kwargs["eval_set"] = eval_set
        self.estimator.fit(X_tr, y_tr, **fit_kwargs)
        raw_best = int(getattr(self.estimator, "best_iteration_", 0) or 0)
        self.best_iteration_ = raw_best or self.n_estimators
        self.evals_result_ = getattr(self.estimator, "evals_result_", {})
        self.is_fitted = True
        return self

    def predict_proba(self, X: Frame) -> np.ndarray:
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted before predicting.")
        probabilities = self.estimator.predict_proba(self._prepare_dataframe(X))
        classes = np.asarray(self.estimator.classes_)
        positive = np.flatnonzero(classes == 1)
        if not len(positive):
            raise RuntimeError("The fitted estimator has no positive class (1).")
        return probabilities[:, int(positive[0])].astype(np.float64)

    def predict(self, X: Frame, threshold: float = 0.5) -> np.ndarray:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    def evaluate(self, X: Frame, y: Labels, dataset_name: Optional[str] = None) -> Dict[str, float]:
        """Optional metric helper; training entry points never call it automatically."""
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )

        y_true = self._labels(y)
        y_prob = self.predict_proba(X)
        prefix = f"{dataset_name.lower()}_" if dataset_name else ""
        two_classes = len(np.unique(y_true)) > 1
        return {
            f"{prefix}roc_auc": float(roc_auc_score(y_true, y_prob)) if two_classes else 0.5,
            f"{prefix}log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
            f"{prefix}pr_auc": (
                float(average_precision_score(y_true, y_prob))
                if two_classes
                else float(np.mean(y_true))
            ),
            f"{prefix}brier_score": float(brier_score_loss(y_true, y_prob)),
        }

    def get_feature_importance(
        self, importance_type: str = "gain", top_k: Optional[int] = None
    ) -> pl.DataFrame:
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted first.")
        booster = self.estimator.booster_
        result = pl.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance": booster.feature_importance(importance_type=importance_type),
            }
        ).sort("importance", descending=True)
        return result.head(top_k) if top_k is not None and top_k > 0 else result

    def save(self, filepath: Union[str, Path]) -> None:
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted before saving.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LightGBMModel":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Artifact at {path} is not a {cls.__name__}.")
        return instance


LightGBMCTRModel = LightGBMModel
