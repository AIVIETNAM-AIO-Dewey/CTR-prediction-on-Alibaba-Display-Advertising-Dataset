"""
LightGBM Model for CTR Prediction.
Self-contained, high-performance gradient boosted decision trees with native Polars support,
histogram binning, early stopping, and end-to-end configuration execution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import yaml

# Filter harmless lightgbm scikit-learn API deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", message=".*LGBMDeprecationWarning.*")

logger = logging.getLogger(__name__)




class LightGBMModel:
    """
    LightGBM Model for Click-Through Rate (CTR) Prediction.

    Key Capabilities:
    - Native Polars DataFrame processing with zero Pandas memory overhead.
    - Automatic non-negative integer hashing for strings/categoricals (avoids Arrow dictionary crashes).
    - Validation-based Early Stopping monitoring LogLoss and ROC-AUC.
    - Built-in evaluate(X, y) calculating LogLoss and ROC-AUC.
    - End-to-end execution directly from config: `LightGBMModel.fit_from_config(...)`.
    - Feature importance analysis returning Polars DataFrames.
    - Serialization (save / load) and YAML configuration loader (from_config).
    """

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
        config: Optional[Dict[str, Any]] = None,
        **extra_kwargs: Any,
    ):
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
        self.config = config or {}
        self.extra_kwargs = extra_kwargs

        self.feature_names: List[str] = []
        self.estimator: Optional[lgb.LGBMClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.evals_result_: Dict[str, Any] = {}
        self.is_fitted: bool = False

    @classmethod
    def load_dataset(
        cls,
        config_path_or_dict: Union[str, Path, Dict[str, Any]],
        sample_size: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        data_dir: Optional[Union[str, Path]] = None,
        use_fe: Optional[bool] = None,
    ) -> Tuple[pl.DataFrame, pl.Series, Optional[pl.DataFrame], Optional[pl.Series]]:
        """
        Load train and validation splits as Polars DataFrames using config and auto-path resolution.
        """
        if isinstance(config_path_or_dict, (str, Path)):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = dict(config_path_or_dict)

        from src.models.train import load_dataset_from_config

        dataset = load_dataset_from_config(
            config=cfg,
            processed_dir=data_dir,
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
        """Instantiate LightGBMModel directly from a YAML config with optional keyword overrides."""
        if isinstance(config_path_or_dict, (str, Path)):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = dict(config_path_or_dict)

        params = dict(cfg.get("params", {}))
        features = cfg.get("features", {})
        data_cfg = cfg.get("data", {})

        cat_cols = features.get("categorical", [])
        num_cols = features.get("numeric", [])
        drop_cols = features.get("drop_features", [])

        cat_cols = [c for c in cat_cols if c not in drop_cols]
        num_cols = [c for c in num_cols if c not in drop_cols]

        seed = data_cfg.get("random_seed", 42)
        params.setdefault("random_state", seed)

        # Apply any explicit kwargs overrides
        params.update(kwargs)

        return cls(
            categorical_features=cat_cols,
            numeric_features=num_cols,
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
        """
        End-to-End Execution for Kaggle or Local:
        Loads data, instantiates model with config overrides, fits, evaluates, and optionally saves.
        """
        if isinstance(config_path_or_dict, (str, Path)):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = dict(config_path_or_dict)

        # 1. Load data
        X_train, y_train, X_val, y_val = cls.load_dataset(
            config_path_or_dict=cfg,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
            data_dir=data_dir,
            use_fe=kwargs.get("use_fe"),
        )

        logger.info(
            f"[{cls.__name__}] Train: {X_train.shape[0]:,} rows x {X_train.shape[1]} cols | "
            f"Val: {X_val.shape[0] if X_val is not None else 0:,} rows"
        )

        # 2. Build model with parameter overrides
        model = cls.from_config(cfg, **kwargs)

        # 3. Fit
        start = time.time()
        model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        elapsed = time.time() - start
        logger.info(f"[{cls.__name__}] Training finished in {elapsed:.2f}s.")

        # 4. Evaluate
        metrics: Dict[str, float] = {}
        if X_val is not None and y_val is not None:
            metrics = model.evaluate(X_val, y_val)
            logger.info(f"[{cls.__name__}] Validation Metrics: {metrics}")

        # 5. Save Artifact
        if save_artifact:
            paths_cfg = cfg.get("paths", {})
            out_dir = Path(models_dir or paths_cfg.get("models_dir", "models"))
            out_dir.mkdir(parents=True, exist_ok=True)
            basename = paths_cfg.get("model_basename", "lightgbm")
            use_fe = cfg.get("data", {}).get("use_fe", True)
            save_path = out_dir / f"{basename}{'_fe' if use_fe else '_baseline'}.joblib"
            model.save(save_path)

        return model, metrics

    def _prepare_dataframe(
        self,
        df: Union[pl.DataFrame, np.ndarray],
        cat_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Format Polars DataFrame for LightGBM training.
        - Hashes strings/categoricals into non-negative integer codes (avoids Arrow dictionary errors).
        - Replaces negative sentinel missing values (-1) with null.
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        elif isinstance(df, np.ndarray):
            df = pl.DataFrame(df, schema=self.feature_names)
        elif not isinstance(df, pl.DataFrame):
            raise TypeError("Expected Polars DataFrame, Pandas DataFrame, or numpy ndarray.")

        active_cats = set(cat_cols or self.categorical_features or [])
        exprs = []
        for col in df.columns:
            dtype = df.schema.get(col)
            if dtype in (pl.Utf8, pl.String, pl.Categorical):
                # Hash string categories into positive 31-bit integers for LightGBM
                exprs.append(
                    (pl.col(col).cast(pl.String).hash(seed=self.random_state) % (2**31 - 1))
                    .cast(pl.Int32)
                    .alias(col)
                )
            elif dtype == pl.Boolean:
                exprs.append(pl.col(col).cast(pl.Int8).alias(col))
            elif col in active_cats and dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64
            ):
                exprs.append(
                    pl.when(pl.col(col) < 0)
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        if exprs:
            df = df.with_columns(exprs)

        return df

    def fit(
        self,
        X: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pl.Series, np.ndarray]] = None,
        X_train: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_train: Optional[Union[pl.Series, np.ndarray]] = None,
        X_val: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pl.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50,
        **kwargs: Any,
    ) -> "LightGBMModel":
        """
        Fit LightGBM model. Accepts either (X, y) or (X_train, y_train, X_val, y_val).
        """
        if X is None and X_train is not None:
            X = X_train
        if y is None and y_train is not None:
            y = y_train

        if X is None or y is None:
            raise ValueError("Training features and labels must be provided.")

        if isinstance(X, pl.DataFrame):
            self.feature_names = list(X.columns)
        elif hasattr(X, "shape") and len(self.feature_names) != X.shape[1]:
            self.feature_names = [f"f_{i}" for i in range(X.shape[1])]

        X_tr = self._prepare_dataframe(X, self.categorical_features)
        y_tr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y).ravel()

        active_cat_cols = [c for c in self.categorical_features if c in X_tr.columns]

        eval_set = None
        callbacks: List[Any] = []

        if verbose_eval and verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))

        if X_val is not None and y_val is not None:
            X_va = self._prepare_dataframe(X_val, self.categorical_features)
            y_va = y_val.to_numpy() if isinstance(y_val, pl.Series) else np.asarray(y_val).ravel()
            eval_set = [(X_tr, y_tr), (X_va, y_va)]

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
        }
        if self.extra_kwargs:
            params.update(self.extra_kwargs)

        self.estimator = lgb.LGBMClassifier(**params)

        logger.info(
            f"[{self.model_name}] Training started: {X_tr.shape[0]:,} samples, "
            f"{X_tr.shape[1]} features, max_trees={self.n_estimators}."
        )

        self.estimator.fit(
            X_tr,
            y_tr,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=active_cat_cols if active_cat_cols else "auto",
            **kwargs,
        )

        self.best_iteration_ = getattr(self.estimator, "best_iteration_", None)
        self.evals_result_ = getattr(self.estimator, "evals_result_", {})
        self.is_fitted = True

        logger.info(
            f"[{self.model_name}] Training completed. Best Iteration: {self.best_iteration_}"
        )
        return self

    def predict_proba(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict positive click probabilities (1D float array in [0.0, 1.0])."""
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted before predicting.")

        X_df = self._prepare_dataframe(X, self.categorical_features)
        proba = self.estimator.predict_proba(X_df)
        return proba[:, 1].astype(np.float64)

    def predict(self, X: Union[pl.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int8)

    def evaluate(
        self,
        X: Union[pl.DataFrame, np.ndarray, pd.DataFrame],
        y: Union[pl.Series, np.ndarray, pd.Series],
        dataset_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute ROC-AUC, LogLoss, PR-AUC, and Brier Score metrics."""
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )

        y_true = y.to_numpy() if isinstance(y, (pl.Series, pd.Series)) else np.asarray(y).ravel()
        y_prob = self.predict_proba(X)

        prefix = f"{dataset_name.lower()}_" if dataset_name else ""
        has_two_classes = len(np.unique(y_true)) > 1
        return {
            f"{prefix}roc_auc": float(roc_auc_score(y_true, y_prob)) if has_two_classes else 0.5,
            f"{prefix}log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
            f"{prefix}pr_auc": float(average_precision_score(y_true, y_prob)) if has_two_classes else float(np.mean(y_true)),
            f"{prefix}brier_score": float(brier_score_loss(y_true, y_prob)),
        }

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: Optional[int] = None,
    ) -> pl.DataFrame:
        """Retrieve sorted feature importance as a native Polars DataFrame."""
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted first.")

        booster = self.estimator.booster_
        scores = booster.feature_importance(importance_type=importance_type)
        names = booster.feature_name()

        df_imp = pl.DataFrame({
            "feature": names,
            "importance": scores,
        }).sort("importance", descending=True)

        if top_k is not None and top_k > 0:
            df_imp = df_imp.head(top_k)

        return df_imp

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model artifact to disk."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info(f"[{self.model_name}] Model successfully saved to {save_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LightGBMModel":
        """Load serialized model artifact from disk."""
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at: {load_path}")
        instance = joblib.load(load_path)
        logger.info(f"[{instance.model_name}] Model successfully loaded from {load_path}")
        return instance


# Alias for backward compatibility
LightGBMCTRModel = LightGBMModel
