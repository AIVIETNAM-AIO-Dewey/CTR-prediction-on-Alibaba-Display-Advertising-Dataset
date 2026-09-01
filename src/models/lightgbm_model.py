"""
LightGBM Model Wrapper for CTR Prediction.
Implements high-performance gradient boosted decision trees with native categorical support,
histogram binning, early stopping, and feature importance diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from src.models.base_model import BaseCTRModel

logger = logging.getLogger(__name__)


class LightGBMCTRModel(BaseCTRModel):
    """
    LightGBM Model Wrapper for Click-Through Rate (CTR) Prediction.
    Key Capabilities:
    - Native categorical feature handling (optimal histogram-based split finding).
    - Validation-based Early Stopping monitoring LogLoss and ROC-AUC.
    - Feature importance analysis (Gain and Split counts).
    - Compact serialization and artifact restoration.
    """

    def __init__(
        self,
        objective: str = "binary",
        metric: Union[str, List[str]] = ("binary_logloss", "auc"),
        boosting_type: str = "gbdt",
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_estimators: int = 1000,
        scale_pos_weight: float = 1.0,
        categorical_features: Optional[List[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LightGBMCTRModel.

        Args:
            objective: Optimization objective ('binary').
            metric: Evaluation metric(s) for validation monitoring.
            boosting_type: Gradient boosting variant ('gbdt', 'dart', 'goss').
            learning_rate: Shrinkage rate for tree updates.
            num_leaves: Maximum tree leaves for base learners.
            max_depth: Maximum tree depth limit (-1 for unlimited).
            min_child_samples: Minimum data in one leaf to prevent overfitting.
            subsample: Row subsampling fraction per iteration.
            subsample_freq: Frequency for subsampling.
            colsample_bytree: Feature subsampling fraction per tree.
            reg_alpha: L1 regularization term on weights.
            reg_lambda: L2 regularization term on weights.
            n_estimators: Maximum number of boosting trees.
            scale_pos_weight: Weight of positive class to address class imbalance.
            categorical_features: Explicit list of categorical column names.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel CPU threads.
            config: Additional hyperparameter dictionary overriding defaults.
        """
        super().__init__(model_name="LightGBM", config=config)

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
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimator: Optional[lgb.LGBMClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.evals_result_: Dict[str, Any] = {}

    def _prepare_dataframe(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        cat_cols: List[str],
    ) -> pd.DataFrame:
        """
        Cast categorical columns to pandas 'category' dtype for native LightGBM handling.

        Args:
            df: Input feature DataFrame.
            cat_cols: List of column names to treat as categorical.

        Returns:
            pd.DataFrame: Formatted DataFrame.
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df_out = df.copy()
        for col in cat_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].astype("category")

        return df_out

    def fit(
        self,
        X_train: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
        y_train: Union[pd.Series, pl.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, pl.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50,
        **kwargs: Any,
    ) -> "LightGBMCTRModel":
        """
        Fit LightGBM model with optional validation and early stopping.

        Args:
            X_train: Training features.
            y_train: Training labels (0 or 1).
            X_val: Validation features for early stopping.
            y_val: Validation labels.
            early_stopping_rounds: Early stopping patience.
            verbose_eval: Logging interval for boosting rounds.
            **kwargs: Additional parameters passed to LGBMClassifier.fit().

        Returns:
            self: The fitted model.
        """
        if isinstance(y_train, pl.Series):
            y_train = y_train.to_numpy()
        if isinstance(y_val, pl.Series):
            y_val = y_val.to_numpy()

        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.to_pandas()

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            cat_cols = self.categorical_features or [
                c for c in self.feature_names if c != "price"
            ]
        else:
            raise TypeError("X_train must be a pandas DataFrame or Polars DataFrame.")

        X_train_df = self._prepare_dataframe(X_train, cat_cols)

        # Initialize LGBMClassifier
        self.estimator = lgb.LGBMClassifier(
            objective=self.objective,
            boosting_type=self.boosting_type,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            importance_type="gain",
        )

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            X_val_df = self._prepare_dataframe(X_val, cat_cols)
            eval_set = [(X_train_df, y_train), (X_val_df, y_val)]
            eval_names = ["train", "val"]

            if early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            if verbose_eval > 0:
                callbacks.append(lgb.log_evaluation(period=verbose_eval))
        else:
            eval_names = None

        logger.info(
            f"Training LightGBM on {len(X_train_df):,} samples with {len(self.feature_names)} features "
            f"({len(cat_cols)} categorical, native handling enabled)..."
        )

        self.estimator.fit(
            X_train_df,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric=self.metric,
            categorical_feature=cat_cols,
            callbacks=callbacks,
            **kwargs,
        )

        self.is_fitted = True
        self.best_iteration_ = getattr(self.estimator, "best_iteration_", self.n_estimators)
        self.evals_result_ = getattr(self.estimator, "evals_result_", {})

        logger.info(f"✅ LightGBM training complete. Best iteration: {self.best_iteration_}")

        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val, dataset_name="Validation")

        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict click probability for input samples.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: 1D array of predicted click probabilities.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Model is not fitted. Call .fit() before predicting.")

        cat_cols = self.categorical_features or [
            c for c in self.feature_names if c != "price"
        ]
        X_eval = self._prepare_dataframe(X, cat_cols)

        probas = self.estimator.predict_proba(X_eval)
        return probas[:, 1]

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract feature importances from the fitted LightGBM model.

        Args:
            importance_type: 'gain' (total split gain) or 'split' (number of splits).
            top_k: Optional limit on number of top features to return.

        Returns:
            pd.DataFrame: Table of feature names and relative importances.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Model is not fitted.")

        raw_importance = self.estimator.booster_.feature_importance(importance_type=importance_type)
        total = np.sum(raw_importance) if np.sum(raw_importance) > 0 else 1.0

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": raw_importance,
            "relative_importance_%": (raw_importance / total) * 100.0,
        }).sort_values(by="importance", ascending=False)

        if top_k is not None:
            return df_imp.head(top_k)
        return df_imp

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serialize LightGBM model and configuration.

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
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Saved LightGBM model to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LightGBMCTRModel":
        """
        Deserialize saved LightGBM model.

        Args:
            filepath: Path to serialized artifact.

        Returns:
            LightGBMCTRModel: Reconstructed instance.
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
        instance.is_fitted = payload.get("is_fitted", False)
        logger.info(f"Loaded LightGBM model from {path}")
        return instance
