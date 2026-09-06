"""Logistic-regression wrapper used by the CTR training pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

Frame = Union[pl.DataFrame, pd.DataFrame, np.ndarray]
Labels = Union[pl.Series, pd.Series, np.ndarray]


class LogisticRegressionModel:
    """Binary linear CTR model with persisted sparse preprocessing."""

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 200,
        use_sgd: bool = False,
        alpha: float = 1e-4,
        class_weight: Any = None,
        random_state: int = 42,
        n_jobs: int = -1,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        **extra_kwargs: Any,
    ) -> None:
        self.model_name = "LogisticRegression"
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.use_sgd = use_sgd or solver == "sgd"
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.categorical_features = list(categorical_features or [])
        self.numeric_features = list(numeric_features or [])
        self.config = config or {}
        self.extra_kwargs = dict(extra_kwargs)

        self.feature_names: List[str] = []
        self.active_categorical_features_: List[str] = []
        self.active_numeric_features_: List[str] = []
        self.numeric_medians_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None
        self.ohe_: Optional[OneHotEncoder] = None
        self.estimator: Optional[Union[LogisticRegression, SGDClassifier]] = None
        self.transformed_feature_names_: List[str] = []
        self.best_iteration_ = 0
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
    ) -> "LogisticRegressionModel":
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
        config_path_or_dict: Union[str, Path, Dict[str, Any]] = "configs/logistic_regression.yaml",
        sample_size: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        data_dir: Optional[Union[str, Path]] = None,
        save_artifact: bool = True,
        models_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Tuple["LogisticRegressionModel", Dict[str, float]]:
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
            basename = paths.get("model_basename", "logistic_regression")
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

    def _split_feature_cols(self, frame: pl.DataFrame) -> Tuple[List[str], List[str]]:
        configured_cats = [c for c in self.categorical_features if c in frame.columns]
        configured_nums = [
            c for c in self.numeric_features if c in frame.columns and c not in configured_cats
        ]
        cats = list(configured_cats)
        nums = list(configured_nums)
        for column, dtype in frame.schema.items():
            if column in cats or column in nums:
                continue
            if dtype in (pl.String, pl.Categorical, pl.Enum, pl.Object):
                cats.append(column)
            else:
                nums.append(column)
        return cats, nums

    def _numeric_array(self, frame: pl.DataFrame, *, fitting: bool) -> np.ndarray:
        expressions = [
            pl.col(column).cast(pl.Float64, strict=False).alias(column)
            for column in self.active_numeric_features_
        ]
        values = np.asarray(frame.select(expressions).to_numpy(), dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        if fitting:
            medians = []
            for column in range(values.shape[1]):
                finite = values[np.isfinite(values[:, column]), column]
                medians.append(float(np.median(finite)) if len(finite) else 0.0)
            self.numeric_medians_ = np.asarray(medians, dtype=np.float64)
        if self.numeric_medians_ is None:
            raise RuntimeError("Numeric imputer was not fitted.")
        missing_rows, missing_cols = np.where(np.isnan(values))
        values[missing_rows, missing_cols] = self.numeric_medians_[missing_cols]
        return values

    def _transform_features(self, data: Frame, *, fitting: bool) -> sp.csr_matrix:
        frame = self._to_frame(data, fitting=fitting)
        if fitting:
            (
                self.active_categorical_features_,
                self.active_numeric_features_,
            ) = self._split_feature_cols(frame)

        parts: List[sp.spmatrix] = []
        names: List[str] = []
        if self.active_numeric_features_:
            numeric = self._numeric_array(frame, fitting=fitting)
            if fitting:
                self.scaler_ = StandardScaler(with_mean=False)
                transformed_numeric = self.scaler_.fit_transform(numeric)
            else:
                if self.scaler_ is None:
                    raise RuntimeError("StandardScaler was not fitted.")
                transformed_numeric = self.scaler_.transform(numeric)
            parts.append(sp.csr_matrix(transformed_numeric))
            names.extend(self.active_numeric_features_)

        if self.active_categorical_features_:
            expressions = [
                pl.col(column)
                .cast(pl.String, strict=False)
                .fill_null("__MISSING__")
                .alias(column)
                for column in self.active_categorical_features_
            ]
            categorical = frame.select(expressions).to_numpy()
            if fitting:
                self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                transformed_categorical = self.ohe_.fit_transform(categorical)
            else:
                if self.ohe_ is None:
                    raise RuntimeError("OneHotEncoder was not fitted.")
                transformed_categorical = self.ohe_.transform(categorical)
            parts.append(transformed_categorical)
            names.extend(
                self.ohe_.get_feature_names_out(
                    self.active_categorical_features_
                ).tolist()
            )

        if not parts:
            raise ValueError("No features available to transform.")
        if fitting:
            self.transformed_feature_names_ = names
        return parts[0].tocsr() if len(parts) == 1 else sp.hstack(parts, format="csr")

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
        **kwargs: Any,
    ) -> "LogisticRegressionModel":
        X = X_train if X is None else X
        y = y_train if y is None else y
        if X is None or y is None:
            raise ValueError("Training features and labels must be provided.")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")

        X_csr = self._transform_features(X, fitting=True)
        y_train_array = self._labels(y)
        if X_csr.shape[0] != len(y_train_array):
            raise ValueError("Training features and labels have different row counts.")
        if X_val is not None and y_val is not None:
            validation = self._to_frame(X_val, fitting=False)
            if len(validation) != len(self._labels(y_val)):
                raise ValueError("Validation features and labels have different row counts.")

        if self.use_sgd:
            params: Dict[str, Any] = {
                "loss": "log_loss",
                "penalty": self.penalty,
                "alpha": self.alpha,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                **self.extra_kwargs,
            }
            self.estimator = SGDClassifier(**params)
        else:
            params = {
                "C": self.C,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                **self.extra_kwargs,
            }
            # l2 is sklearn's default. Omitting it also avoids the 1.8+ deprecation
            # warning while remaining compatible with the project's sklearn>=1.3.
            if self.penalty != "l2":
                params["penalty"] = self.penalty
            self.estimator = LogisticRegression(**params)

        self.estimator.fit(X_csr, y_train_array, **kwargs)
        iterations = np.asarray(getattr(self.estimator, "n_iter_", [0])).ravel()
        self.best_iteration_ = int(iterations.max()) if len(iterations) else 0
        self.is_fitted = True
        return self

    def predict_proba(self, X: Frame) -> np.ndarray:
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted before predicting.")
        matrix = self._transform_features(X, fitting=False)
        classes = np.asarray(self.estimator.classes_)
        positive = np.flatnonzero(classes == 1)
        if not len(positive):
            raise RuntimeError("The fitted estimator has no positive class (1).")
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(matrix)[:, int(positive[0])].astype(np.float64)
        decision = self.estimator.decision_function(matrix)
        return (1.0 / (1.0 + np.exp(-decision))).astype(np.float64)

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

    def get_feature_importance(self, top_k: Optional[int] = None) -> pl.DataFrame:
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted first.")
        coefficients = self.estimator.coef_.ravel()
        result = pl.DataFrame(
            {
                "feature": self.transformed_feature_names_,
                "coefficient": coefficients,
                "abs_importance": np.abs(coefficients),
            }
        ).sort("abs_importance", descending=True)
        return result.head(top_k) if top_k is not None and top_k > 0 else result

    def save(self, filepath: Union[str, Path]) -> None:
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted before saving.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LogisticRegressionModel":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Artifact at {path} is not a {cls.__name__}.")
        return instance


LogisticRegressionCTRModel = LogisticRegressionModel
