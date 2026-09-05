"""
XGBoost Model Wrapper for CTR Prediction.
Implements histogram-based gradient boosted trees with native categorical support
(`enable_categorical=True`), train-fitted category dictionaries for leak-free
train/val/test alignment, early stopping, and feature importance diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostCTRModel:
    """
    XGBoost Model Wrapper for Click-Through Rate (CTR) Prediction.

    Key Capabilities:
    - Native categorical splits via `enable_categorical` + `tree_method='hist'`,
      avoiding one-hot explosion on high-cardinality ad identifiers.
    - Category dictionaries fitted on the training partition only and frozen onto
      val/test; unseen levels are routed to the model's missing-value branch.
    - Validation-based early stopping monitoring LogLoss and ROC-AUC.
    - Feature importance analysis returning Polars DataFrames.
    - Compact serialization and artifact restoration.
    """

    def __init__(
        self,
        objective: str = "binary:logistic",
        eval_metric: Union[str, List[str]] = ("logloss", "auc"),
        tree_method: str = "hist",
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        min_child_weight: float = 10.0,
        gamma: float = 0.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        max_bin: int = 256,
        max_cat_to_onehot: int = 8,
        max_cat_threshold: int = 64,
        grow_policy: str = "depthwise",
        categorical_features: Optional[List[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        device: str = "cpu",
        verbosity: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize XGBoostCTRModel.

        Args:
            objective: Optimization objective ('binary:logistic').
            eval_metric: Evaluation metric(s); early stopping monitors the last one.
            tree_method: Tree construction algorithm ('hist' required for categoricals).
            n_estimators: Maximum number of boosting rounds.
            learning_rate: Shrinkage rate (eta) for tree updates.
            max_depth: Maximum tree depth.
            min_child_weight: Minimum sum of instance hessian in a child (regularizer).
            gamma: Minimum loss reduction required for a further split.
            subsample: Row subsampling fraction per boosting round.
            colsample_bytree: Feature subsampling fraction per tree.
            reg_alpha: L1 regularization term on weights.
            reg_lambda: L2 regularization term on weights.
            scale_pos_weight: Weight of the positive class to address class imbalance.
            max_bin: Number of histogram bins for numerical features.
            max_cat_to_onehot: Categories below this cardinality use one-hot splits.
            max_cat_threshold: Maximum categories considered per categorical split.
            grow_policy: 'depthwise' or 'lossguide' tree growth.
            categorical_features: Explicit list of categorical column names.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel CPU threads (-1 for all cores).
            device: 'cpu' or 'cuda'.
            verbosity: Engine verbosity level.
            config: Additional hyperparameter dictionary overriding defaults.
        """
        self.model_name = "XGBoost"
        self.config = config or {}
        self.categorical_features = list(categorical_features or [])
        self.numeric_features: List[str] = []
        self.feature_names: List[str] = []
        self.is_fitted = False

        self.objective = objective
        self.eval_metric = list(eval_metric) if isinstance(eval_metric, (list, tuple)) else [eval_metric]
        self.tree_method = tree_method
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.max_bin = max_bin
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.grow_policy = grow_policy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.verbosity = verbosity

        self.estimator: Optional[xgb.XGBClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.evals_result_: Dict[str, Any] = {}
        self.active_cat_features_: List[str] = []
        self.category_dtypes_: Dict[str, pd.CategoricalDtype] = {}

    # ------------------------------------------------------------------ #
    # Data Preparation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _string_columns(df: Union[pl.DataFrame, pd.DataFrame]) -> List[str]:
        """Return string-like columns that must use XGBoost categorical support."""
        if isinstance(df, pd.DataFrame):
            return [
                col
                for col in df.columns
                if (
                    pd.api.types.is_object_dtype(df[col])
                    or pd.api.types.is_string_dtype(df[col])
                    or isinstance(df[col].dtype, pd.CategoricalDtype)
                )
            ]

        string_dtypes = {pl.Utf8, pl.String, pl.Categorical, pl.Enum, pl.Object}
        return [col for col, dtype in df.schema.items() if dtype in string_dtypes]

    @staticmethod
    def _to_pandas(df: Union[pl.DataFrame, np.ndarray, pd.DataFrame], schema: List[str]) -> pd.DataFrame:
        """Normalize any supported input container into a pandas DataFrame."""
        if isinstance(df, pd.DataFrame):
            return df
        if isinstance(df, np.ndarray):
            return pd.DataFrame(df, columns=schema)
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        raise TypeError("Expected Polars DataFrame, pandas DataFrame or numpy ndarray.")

    def _fit_category_dtypes(self, X_pd: pd.DataFrame) -> None:
        """Freeze the category dictionary of each categorical column on the train partition."""
        self.category_dtypes_ = {}
        for col in self.active_cat_features_:
            categories = pd.Index(X_pd[col].dropna().astype(str).unique())
            self.category_dtypes_[col] = pd.CategoricalDtype(categories=categories)

    def _prepare_dataframe(
        self,
        df: Union[pl.DataFrame, np.ndarray, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Convert a feature matrix into the pandas layout XGBoost expects.
        - Categorical columns are cast to the train-fitted `category` dtype so that
          internal category codes stay aligned across train/val/test; levels unseen
          in training become NaN and follow the missing-value branch.
        - Remaining columns are cast to float32.

        Args:
            df: Input feature matrix.

        Returns:
            pd.DataFrame: XGBoost-ready feature matrix with columns in training order.
        """
        X_pd = self._to_pandas(df, self.feature_names)

        if self.feature_names:
            missing = [c for c in self.feature_names if c not in X_pd.columns]
            if missing:
                raise ValueError(f"Missing expected feature columns: {missing}")
            X_pd = X_pd.loc[:, self.feature_names]
        else:
            X_pd = X_pd.copy()

        prepared = {}
        for col in X_pd.columns:
            series = X_pd[col]
            if col in self.category_dtypes_:
                # pandas StringDtype preserves nulls as <NA>; casting through Python str would
                # turn them into the literal category "nan" and hide missing-value routing.
                values = series.astype("string")
                known = self.category_dtypes_[col].categories
                values = values.where(values.isin(known), pd.NA)
                prepared[col] = values.astype(self.category_dtypes_[col])
            elif col in self.active_cat_features_:
                prepared[col] = series.astype("string").astype("category")
            elif (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                raise ValueError(
                    f"String feature '{col}' is not declared as categorical. "
                    "Add it to categorical_features."
                )
            else:
                prepared[col] = pd.to_numeric(series, errors="coerce").astype(np.float32)

        return pd.DataFrame(prepared, index=X_pd.index)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: Union[pl.DataFrame, np.ndarray],
        y_train: Union[pl.Series, np.ndarray],
        X_val: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pl.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 100,
        verbose_eval: int = 50,
        **kwargs: Any,
    ) -> "XGBoostCTRModel":
        """
        Fit XGBoost with optional validation-based early stopping.

        Args:
            X_train: Training features (Polars DataFrame preferred).
            y_train: Training labels (0 or 1).
            X_val: Validation features for early stopping.
            y_val: Validation labels.
            early_stopping_rounds: Early stopping patience (0 to disable).
            verbose_eval: Logging interval for boosting rounds (0 to silence).
            **kwargs: Additional parameters passed to XGBClassifier.fit().

        Returns:
            self: The fitted model.
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be omitted.")

        if isinstance(y_train, (pl.Series, pd.Series)):
            y_train = y_train.to_numpy()
        if isinstance(y_val, (pl.Series, pd.Series)):
            y_val = y_val.to_numpy()
        y_train = np.asarray(y_train).ravel()
        if y_val is not None:
            y_val = np.asarray(y_val).ravel()

        if isinstance(X_train, (pl.DataFrame, pd.DataFrame)):
            self.feature_names = list(X_train.columns)
        elif isinstance(X_train, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            raise TypeError(
                "X_train must be a Polars DataFrame, pandas DataFrame, or numpy ndarray."
            )

        configured_cats = [c for c in self.categorical_features if c in self.feature_names]
        detected_cats = []
        if isinstance(X_train, (pl.DataFrame, pd.DataFrame)):
            detected_cats = [c for c in self._string_columns(X_train) if c in self.feature_names]
        self.active_cat_features_ = list(dict.fromkeys(configured_cats + detected_cats))

        # Fit category dictionaries on train only, then reuse them for val/test
        train_raw = self._to_pandas(X_train, self.feature_names).loc[:, self.feature_names]
        self._fit_category_dtypes(train_raw)
        X_train_df = self._prepare_dataframe(train_raw)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train_df, y_train), (self._prepare_dataframe(X_val), y_val)]

        use_early_stopping = eval_set is not None and early_stopping_rounds > 0

        self.estimator = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            max_bin=self.max_bin,
            max_cat_to_onehot=self.max_cat_to_onehot,
            max_cat_threshold=self.max_cat_threshold,
            grow_policy=self.grow_policy,
            enable_categorical=True,
            early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            device=self.device,
            verbosity=self.verbosity,
        )

        logger.info(
            f"Training XGBoost on {len(X_train_df):,} samples with {len(self.feature_names)} features "
            f"({len(self.active_cat_features_)} categorical, max_depth={self.max_depth}, "
            f"n_estimators={self.n_estimators})..."
        )

        self.estimator.fit(
            X_train_df,
            y_train,
            eval_set=eval_set,
            verbose=verbose_eval if verbose_eval and verbose_eval > 0 else False,
            **kwargs,
        )

        self.is_fitted = True
        self.best_iteration_ = getattr(self.estimator, "best_iteration", None)
        if self.best_iteration_ is None:
            # XGBoost uses zero-based iteration indices; the last tree is n_estimators - 1.
            self.best_iteration_ = max(self.n_estimators - 1, 0)
        self.evals_result_ = self.estimator.evals_result() if eval_set is not None else {}

        logger.info(f"XGBoost training complete. Best iteration: {self.best_iteration_}")

        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_proba(
        self,
        X: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict click probability for input samples.

        Args:
            X: Feature matrix (Polars DataFrame preferred).

        Returns:
            np.ndarray: 1D array of predicted click probabilities.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Model is not fitted. Call .fit() before predicting.")

        X_eval = self._prepare_dataframe(X)
        return self.estimator.predict_proba(X_eval)[:, 1]

    def predict(
        self,
        X: Union[pl.DataFrame, pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary predictions using a configurable probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    # ------------------------------------------------------------------ #
    # Diagnostics & Persistence
    # ------------------------------------------------------------------ #
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract feature importances from the fitted XGBoost model as a Polars DataFrame.

        Args:
            importance_type: 'gain', 'weight', 'cover', 'total_gain' or 'total_cover'.
            top_k: Optional limit on number of top features to return.

        Returns:
            pl.DataFrame: Table of feature names and relative importances.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Model is not fitted.")

        score_map = self.estimator.get_booster().get_score(importance_type=importance_type)
        raw_importance = np.array(
            [float(score_map.get(name, 0.0)) for name in self.feature_names], dtype=np.float64
        )
        total = float(np.sum(raw_importance)) if np.sum(raw_importance) > 0 else 1.0

        df_imp = pl.DataFrame({
            "feature": self.feature_names,
            "importance": raw_importance,
            "relative_importance_%": (raw_importance / total) * 100.0,
        }).sort(by="importance", descending=True)

        if top_k is not None:
            return df_imp.head(top_k)
        return df_imp

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serialize XGBoost model, category dictionaries, and configuration.

        Args:
            filepath: Destination file path.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "estimator": self.estimator,
            "best_iteration": self.best_iteration_,
            "config": self.config,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "active_cat_features": self.active_cat_features_,
            "category_dtypes": self.category_dtypes_,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Saved XGBoost model to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "XGBoostCTRModel":
        """
        Deserialize saved XGBoost model.

        Args:
            filepath: Path to serialized artifact.

        Returns:
            XGBoostCTRModel: Reconstructed instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        payload = joblib.load(path)
        instance = cls(
            categorical_features=payload.get("categorical_features"),
            config=payload.get("config"),
        )
        instance.estimator = payload.get("estimator")
        instance.best_iteration_ = payload.get("best_iteration")
        instance.feature_names = payload.get("feature_names", [])
        instance.active_cat_features_ = payload.get("active_cat_features", [])
        instance.category_dtypes_ = payload.get("category_dtypes", {})
        instance.is_fitted = payload.get("is_fitted", False)
        logger.info(f"Loaded XGBoost model from {path}")
        return instance
