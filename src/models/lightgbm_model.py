"""
LightGBM Model Wrapper for CTR Prediction.
Implements high-performance gradient boosted decision trees with native Polars DataFrame support,
histogram binning, early stopping, and feature importance diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.base_model import BaseCTRModel

# Filter harmless lightgbm scikit-learn API deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", message=".*LGBMDeprecationWarning.*")

logger = logging.getLogger(__name__)


class LightGBMCTRModel(BaseCTRModel):
    """
    LightGBM Model Wrapper for Click-Through Rate (CTR) Prediction.
    Key Capabilities:
    - Native Polars DataFrame processing with zero memory duplication.
    - Automatic categorical encoding and null-handling for LightGBM.
    - Validation-based Early Stopping monitoring LogLoss and ROC-AUC.
    - Feature importance analysis returning Polars DataFrames.
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
        min_child_samples: int = 20,
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
        verbose: int = -1,
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
            verbose: Verbosity level (-1 to suppress noisy engine warnings).
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
        self.verbose = verbose

        self.estimator: Optional[lgb.LGBMClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.evals_result_: Dict[str, Any] = {}

    def _prepare_dataframe(
        self,
        df: Union[pl.DataFrame, np.ndarray],
        cat_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Format Polars DataFrame for LightGBM training.
        - Converts negative sentinel values (-1) in categoricals to nulls.
        - Hashes strings/categoricals into non-negative integer codes.

        Args:
            df: Input feature matrix.
            cat_cols: Optional list of categorical column names.

        Returns:
            pl.DataFrame: Formatted Polars DataFrame.
        """
        if isinstance(df, np.ndarray):
            df = pl.DataFrame(df, schema=self.feature_names)
        elif not isinstance(df, pl.DataFrame):
            raise TypeError("Expected Polars DataFrame or numpy ndarray.")

        active_cats = set(cat_cols or self.categorical_features or [])
        exprs = []
        for col in df.columns:
            dtype = df.schema.get(col)
            if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Object):
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
                # Replace negative sentinel missing values (-1) with null to avoid negative category warning
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
        X_train: Union[pl.DataFrame, np.ndarray],
        y_train: Union[pl.Series, np.ndarray],
        X_val: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pl.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50,
        **kwargs: Any,
    ) -> "LightGBMCTRModel":
        """
        Fit LightGBM model with optional validation and early stopping.

        Args:
            X_train: Training features (Polars DataFrame preferred).
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
            self.feature_names = list(X_train.columns)
            cat_cols = self.categorical_features or [
                c for c in self.feature_names if c != "price"
            ]
        elif isinstance(X_train, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            cat_cols = self.categorical_features or []
        else:
            raise TypeError("X_train must be a Polars DataFrame or numpy ndarray.")

        active_cat_cols = [c for c in cat_cols if c in self.feature_names]
        X_train_df = self._prepare_dataframe(X_train, active_cat_cols)

        # Scale min_child_samples automatically if sample size is small
        min_child = self.min_child_samples
        if len(X_train) < 5000:
            min_child = max(5, min(self.min_child_samples, max(5, int(len(X_train) * 0.02))))

        # Initialize LGBMClassifier
        self.estimator = lgb.LGBMClassifier(
            objective=self.objective,
            boosting_type=self.boosting_type,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=min_child,
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
            verbose=self.verbose,
        )

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            X_val_df = self._prepare_dataframe(X_val, active_cat_cols)
            eval_set = [(X_train_df, y_train), (X_val_df, y_val)]
            eval_names = ["train", "val"]

            if early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            if verbose_eval > 0:
                callbacks.append(lgb.log_evaluation(period=verbose_eval))
        else:
            eval_names = None

        logger.info(
            f"Training LightGBM (Polars) on {len(X_train_df):,} samples with {len(self.feature_names)} features "
            f"({len(active_cat_cols)} categorical features, min_child_samples={min_child})..."
        )

        self.estimator.fit(
            X_train_df,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric=self.metric,
            categorical_feature=active_cat_cols if active_cat_cols else "auto",
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
        probas = self.estimator.predict_proba(X_eval)
        return probas[:, 1]

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract feature importances from the fitted LightGBM model as a Polars DataFrame.

        Args:
            importance_type: 'gain' (total split gain) or 'split' (number of splits).
            top_k: Optional limit on number of top features to return.

        Returns:
            pl.DataFrame: Table of feature names and relative importances.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Model is not fitted.")

        raw_importance = self.estimator.booster_.feature_importance(importance_type=importance_type)
        total = float(np.sum(raw_importance)) if np.sum(raw_importance) > 0 else 1.0

        df_imp = pl.DataFrame({
            "feature": self.feature_names,
            "importance": raw_importance.astype(np.float64),
            "relative_importance_%": ((raw_importance / total) * 100.0).astype(np.float64),
        }).sort(by="importance", descending=True)

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
