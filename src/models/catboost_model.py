"""
CatBoost Model Wrapper for CTR Prediction.
Implements ordered boosting with native high-cardinality categorical handling
(target statistics + automatic feature combinations), early stopping on validation
LogLoss / ROC-AUC, and feature importance diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import catboost as cb
import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.models.base_model import BaseCTRModel

logger = logging.getLogger(__name__)

# Placeholder used for null categorical levels (CatBoost rejects NaN in cat features)
_CAT_NULL_TOKEN = "__NA__"


class CatBoostCTRModel(BaseCTRModel):
    """
    CatBoost Model Wrapper for Click-Through Rate (CTR) Prediction.

    Key Capabilities:
    - Native categorical handling via ordered target statistics, removing the need
      for manual label/target encoding of `cate_id`, `brand`, `customer`, `campaign_id`.
    - Ordered boosting to mitigate target leakage / prediction shift on encoded categories.
    - Validation-based early stopping monitoring LogLoss and ROC-AUC.
    - Feature importance analysis returning Polars DataFrames.
    - Compact serialization and artifact restoration.
    """

    def __init__(
        self,
        loss_function: str = "Logloss",
        eval_metric: str = "AUC",
        iterations: int = 2000,
        learning_rate: float = 0.05,
        depth: int = 8,
        l2_leaf_reg: float = 3.0,
        border_count: int = 128,
        one_hot_max_size: int = 10,
        max_ctr_complexity: int = 2,
        bootstrap_type: str = "Bernoulli",
        subsample: float = 0.8,
        rsm: float = 0.8,
        scale_pos_weight: float = 1.0,
        boosting_type: str = "Plain",
        categorical_features: Optional[List[str]] = None,
        random_state: int = 42,
        thread_count: int = -1,
        task_type: str = "CPU",
        verbose: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CatBoostCTRModel.

        Args:
            loss_function: Optimization objective ('Logloss' for binary CTR).
            eval_metric: Metric monitored for early stopping ('AUC' or 'Logloss').
            iterations: Maximum number of boosting iterations.
            learning_rate: Shrinkage rate for tree updates.
            depth: Depth of the symmetric (oblivious) trees.
            l2_leaf_reg: L2 regularization coefficient on leaf values.
            border_count: Number of splits considered for numerical features.
            one_hot_max_size: Categories below this cardinality use one-hot instead of CTR.
            max_ctr_complexity: Maximum number of features in automatic categorical combinations.
            bootstrap_type: Row sampling scheme ('Bernoulli', 'Bayesian', 'MVS', 'No').
            subsample: Row subsampling fraction (requires Bernoulli / MVS bootstrap).
            rsm: Feature subsampling fraction per split (CPU only).
            scale_pos_weight: Weight of the positive class to address class imbalance.
            boosting_type: 'Plain' (fast, large datasets) or 'Ordered' (less biased, small datasets).
            categorical_features: Explicit list of categorical column names.
            random_state: Random seed for reproducibility.
            thread_count: Number of parallel CPU threads (-1 for all cores).
            task_type: 'CPU' or 'GPU'.
            verbose: Logging interval for boosting rounds (0 to silence).
            config: Additional hyperparameter dictionary overriding defaults.
        """
        super().__init__(model_name="CatBoost", config=config)

        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.one_hot_max_size = one_hot_max_size
        self.max_ctr_complexity = max_ctr_complexity
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.rsm = rsm
        self.scale_pos_weight = scale_pos_weight
        self.boosting_type = boosting_type
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.thread_count = thread_count
        self.task_type = task_type
        self.verbose = verbose

        self.estimator: Optional[cb.CatBoostClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.active_cat_features_: List[str] = []

    # ------------------------------------------------------------------ #
    # Data Preparation
    # ------------------------------------------------------------------ #
    def _prepare_dataframe(
        self,
        df: Union[pl.DataFrame, np.ndarray, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Convert a Polars feature matrix into the pandas layout CatBoost expects.
        - Categorical columns are cast to string with nulls replaced by a stable token.
        - Numerical columns are cast to float32 and keep NaN (CatBoost handles them natively).

        Args:
            df: Input feature matrix.

        Returns:
            pd.DataFrame: CatBoost-ready feature matrix with columns in training order.
        """
        if isinstance(df, np.ndarray):
            df = pl.DataFrame(df, schema=self.feature_names)
        elif isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        elif not isinstance(df, pl.DataFrame):
            raise TypeError("Expected Polars DataFrame, pandas DataFrame or numpy ndarray.")

        if self.feature_names:
            missing = [c for c in self.feature_names if c not in df.columns]
            if missing:
                raise ValueError(f"Missing expected feature columns: {missing}")
            df = df.select(self.feature_names)

        cat_cols = set(self.active_cat_features_ or self.categorical_features or [])

        exprs = []
        for col in df.columns:
            if col in cat_cols:
                exprs.append(
                    pl.col(col).cast(pl.Utf8).fill_null(_CAT_NULL_TOKEN).alias(col)
                )
            else:
                dtype = df.schema.get(col)
                if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum, pl.Object):
                    # Unexpected string column outside the declared categorical list
                    exprs.append(
                        pl.col(col).cast(pl.Utf8).fill_null(_CAT_NULL_TOKEN).alias(col)
                    )
                else:
                    exprs.append(pl.col(col).cast(pl.Float32).alias(col))

        return df.with_columns(exprs).to_pandas()

    def _make_pool(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
    ) -> cb.Pool:
        """Build a CatBoost Pool with categorical feature indices resolved by name."""
        X_pd = self._prepare_dataframe(X)
        cat_indices = [X_pd.columns.get_loc(c) for c in self.active_cat_features_]
        return cb.Pool(data=X_pd, label=y, cat_features=cat_indices)

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
        **kwargs: Any,
    ) -> "CatBoostCTRModel":
        """
        Fit CatBoost with optional validation-based early stopping.

        Args:
            X_train: Training features (Polars DataFrame preferred).
            y_train: Training labels (0 or 1).
            X_val: Validation features for early stopping.
            y_val: Validation labels.
            early_stopping_rounds: Early stopping patience (0 to disable).
            **kwargs: Additional parameters passed to CatBoostClassifier.fit().

        Returns:
            self: The fitted model.
        """
        if isinstance(y_train, pl.Series):
            y_train = y_train.to_numpy()
        if isinstance(y_val, pl.Series):
            y_val = y_val.to_numpy()

        if isinstance(X_train, pl.DataFrame):
            self.feature_names = list(X_train.columns)
        elif isinstance(X_train, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            raise TypeError("X_train must be a Polars DataFrame or numpy ndarray.")

        self.active_cat_features_ = [
            c for c in (self.categorical_features or []) if c in self.feature_names
        ]

        train_pool = self._make_pool(X_train, y_train)

        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = self._make_pool(X_val, y_val)

        params: Dict[str, Any] = {
            "loss_function": self.loss_function,
            "eval_metric": self.eval_metric,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "border_count": self.border_count,
            "one_hot_max_size": self.one_hot_max_size,
            "max_ctr_complexity": self.max_ctr_complexity,
            "bootstrap_type": self.bootstrap_type,
            "boosting_type": self.boosting_type,
            "scale_pos_weight": self.scale_pos_weight,
            "random_seed": self.random_state,
            "thread_count": self.thread_count,
            "task_type": self.task_type,
            "verbose": self.verbose,
            "allow_writing_files": False,
        }

        # `subsample` is only valid for Bernoulli / MVS / Poisson bootstrap schemes
        if self.bootstrap_type in ("Bernoulli", "MVS", "Poisson"):
            params["subsample"] = self.subsample
        # `rsm` (feature subsampling) is unsupported on GPU for non-pairwise losses
        if self.task_type == "CPU":
            params["rsm"] = self.rsm

        self.estimator = cb.CatBoostClassifier(**params)

        logger.info(
            f"Training CatBoost on {train_pool.num_row():,} samples with "
            f"{len(self.feature_names)} features ({len(self.active_cat_features_)} categorical, "
            f"depth={self.depth}, iterations={self.iterations})..."
        )

        self.estimator.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping_rounds if (eval_pool is not None and early_stopping_rounds > 0) else None,
            use_best_model=eval_pool is not None,
            verbose=self.verbose,
            **kwargs,
        )

        self.is_fitted = True
        self.best_iteration_ = self.estimator.get_best_iteration()

        logger.info(f"CatBoost training complete. Best iteration: {self.best_iteration_}")

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

        pool = self._make_pool(X)
        return self.estimator.predict_proba(pool)[:, 1]

    # ------------------------------------------------------------------ #
    # Diagnostics & Persistence
    # ------------------------------------------------------------------ #
    def get_feature_importance(
        self,
        importance_type: str = "PredictionValuesChange",
        top_k: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract feature importances from the fitted CatBoost model as a Polars DataFrame.

        Args:
            importance_type: 'PredictionValuesChange' (default, no data required) or
                'LossFunctionChange' (requires an evaluation Pool).
            top_k: Optional limit on number of top features to return.

        Returns:
            pl.DataFrame: Table of feature names and relative importances.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Model is not fitted.")

        raw_importance = np.asarray(
            self.estimator.get_feature_importance(type=importance_type), dtype=np.float64
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
        Serialize CatBoost model and configuration.

        Args:
            filepath: Destination file path.
        """
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
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Saved CatBoost model to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CatBoostCTRModel":
        """
        Deserialize saved CatBoost model.

        Args:
            filepath: Path to serialized artifact.

        Returns:
            CatBoostCTRModel: Reconstructed instance.
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
        instance.is_fitted = payload.get("is_fitted", False)
        logger.info(f"Loaded CatBoost model from {path}")
        return instance
