"""
Random Forest Model Wrapper for CTR Prediction.
Bagging baseline: scikit-learn RandomForestClassifier over ordinal-encoded categoricals,
with dictionaries fitted on train only and frozen onto val/test, plus Gini feature importances.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# Code assigned to categorical levels never seen during training
_UNSEEN_CODE = -1


class RandomForestCTRModel:
    """
    Random Forest Model Wrapper for Click-Through Rate (CTR) Prediction.

    Serves as the bagging baseline the boosted models are compared against. Unlike CatBoost and
    XGBoost, scikit-learn's forest has no native categorical handling, so categorical columns are
    ordinal-encoded against a dictionary fitted on the training partition; levels unseen later map
    to a single out-of-vocabulary code. Very high-cardinality identifiers should be excluded via
    the config's `features.drop_features` and consumed through their target encodings instead --
    an ordinal code over 240K advertiser IDs carries no usable order for a split.

    Key Capabilities:
    - Bagged trees over bootstrapped rows and a random feature subset per split.
    - Probability output averaged over per-tree leaf frequencies (no early stopping to configure).
    - Gini importance analysis returning Polars DataFrames.
    - Compact serialization and artifact restoration.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        criterion: str = "gini",
        max_depth: Optional[int] = 16,
        min_samples_split: int = 20,
        min_samples_leaf: int = 50,
        max_features: Union[str, int, float] = "sqrt",
        max_samples: Optional[float] = 0.8,
        bootstrap: bool = True,
        oob_score: bool = False,
        class_weight: Optional[str] = None,
        categorical_features: Optional[List[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RandomForestCTRModel.

        Args:
            n_estimators: Number of trees in the forest.
            criterion: Split quality function ('gini' or 'entropy').
            max_depth: Maximum tree depth (None grows until the leaf constraints bite).
            min_samples_split: Minimum samples required to split an internal node.
            min_samples_leaf: Minimum samples per leaf; the main guard on probability noise.
            max_features: Features considered per split ('sqrt', 'log2', an int or a fraction).
            max_samples: Fraction of rows drawn per tree when bootstrap is True.
            bootstrap: Whether to bootstrap rows per tree (required for oob_score).
            oob_score: Whether to compute sklearn's out-of-bag score (accuracy) during fit.
                Off by default: accuracy is uninformative at a ~5% base rate, and metrics are
                Task 4's responsibility.
            class_weight: Class re-weighting ('balanced' or None). None preserves calibration.
            categorical_features: Explicit list of categorical column names.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 for all cores).
            verbose: Verbosity level passed to scikit-learn.
            config: Additional hyperparameter dictionary overriding defaults.
        """
        self.model_name = "RandomForest"
        self.config = config or {}
        self.categorical_features = list(categorical_features or [])
        self.numeric_features: List[str] = []
        self.feature_names: List[str] = []
        self.is_fitted = False

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimator: Optional[RandomForestClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.active_cat_features_: List[str] = []
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.numeric_medians_: Dict[str, float] = {}
        self.oob_score_: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Data Preparation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_polars(
        df: Union[pl.DataFrame, pd.DataFrame, np.ndarray], schema: List[str]
    ) -> pl.DataFrame:
        """Normalize any supported input container into a Polars DataFrame."""
        if isinstance(df, pl.DataFrame):
            return df
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        if isinstance(df, np.ndarray):
            return pl.DataFrame(df, schema=schema)
        raise TypeError("Expected Polars DataFrame, pandas DataFrame, or numpy ndarray.")

    @staticmethod
    def _string_columns(df: Union[pl.DataFrame, pd.DataFrame]) -> List[str]:
        """Return string-like columns that must use the fitted category map."""
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

    def _fit_category_maps(self, X: pl.DataFrame) -> None:
        """Freeze a level -> integer code dictionary per categorical column, from train only."""
        self.category_maps_ = {}
        for col in self.active_cat_features_:
            levels = X.get_column(col).cast(pl.Utf8).unique().drop_nulls().sort().to_list()
            self.category_maps_[col] = {level: code for code, level in enumerate(levels)}

    def _fit_numeric_medians(self, X: pl.DataFrame) -> None:
        """Fit numeric imputation values on train only for sklearn 1.3 compatibility."""
        self.numeric_medians_ = {}
        for col in self.feature_names:
            if col in self.active_cat_features_:
                continue
            values = X.get_column(col).cast(pl.Float64, strict=False)
            median = values.median()
            median_value = float(median) if median is not None and np.isfinite(median) else 0.0
            self.numeric_medians_[col] = median_value

    def _prepare_matrix(self, df: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Convert a feature matrix into the dense float32 array scikit-learn expects.
        - Categorical columns are mapped through the train-fitted dictionary; levels unseen in
          training (and nulls) collapse onto a single out-of-vocabulary code.
        - Remaining columns are cast to float32 and keep NaN, which scikit-learn splits on natively.

        Args:
            df: Input feature matrix.

        Returns:
            np.ndarray: 2D float32 array with columns in training order.
        """
        X = self._to_polars(df, self.feature_names)

        if self.feature_names:
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing expected feature columns: {missing}")
            X = X.select(self.feature_names)

        exprs = []
        for col in X.columns:
            if col in self.category_maps_:
                exprs.append(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .replace_strict(self.category_maps_[col], default=_UNSEEN_CODE)
                    .cast(pl.Float32)
                    .alias(col)
                )
            else:
                median = self.numeric_medians_.get(col, 0.0)
                exprs.append(
                    pl.col(col)
                    .cast(pl.Float32, strict=False)
                    .fill_nan(median)
                    .fill_null(median)
                    .alias(col)
                )

        return X.with_columns(exprs).to_numpy().astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: Union[pl.DataFrame, np.ndarray],
        y_train: Union[pl.Series, np.ndarray],
        X_val: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pl.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "RandomForestCTRModel":
        """
        Fit the forest on the training partition.

        A forest has no sequential boosting rounds, so there is nothing to stop early: `X_val` and
        `y_val` are accepted for interface symmetry with the boosted wrappers but are not used for
        training.

        Args:
            X_train: Training features (Polars DataFrame preferred).
            y_train: Training labels (0 or 1).
            X_val: Ignored; kept for interface compatibility.
            y_val: Ignored; kept for interface compatibility.
            **kwargs: Additional parameters passed to RandomForestClassifier.fit().

        Returns:
            self: The fitted model.
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be omitted.")

        if isinstance(y_train, (pl.Series, pd.Series)):
            y_train = y_train.to_numpy()
        y_train = np.asarray(y_train).ravel()

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

        if X_val is not None:
            logger.info("Random Forest has no early stopping; the validation partition is unused.")

        # Dictionaries come from train only, then stay frozen for val/test.
        X_train_pl = self._to_polars(X_train, self.feature_names)
        self._fit_category_maps(X_train_pl)
        self._fit_numeric_medians(X_train_pl)
        X_train_np = self._prepare_matrix(X_train_pl)

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_samples=self.max_samples if self.bootstrap else None,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score and self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        logger.info(
            f"Training Random Forest on {len(X_train_np):,} samples with "
            f"{len(self.feature_names)} features ({len(self.active_cat_features_)} categorical, "
            f"max_depth={self.max_depth}, n_estimators={self.n_estimators})..."
        )

        self.estimator.fit(X_train_np, y_train, **kwargs)

        self.is_fitted = True
        self.best_iteration_ = max(self.n_estimators - 1, 0)  # No early stopping: every tree is kept
        self.oob_score_ = float(getattr(self.estimator, "oob_score_", np.nan))

        logger.info(
            f"Random Forest training complete. Trees: {self.n_estimators}"
            + (f" | OOB score: {self.oob_score_:.5f}" if np.isfinite(self.oob_score_) else "")
        )

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

        probabilities = self.estimator.predict_proba(self._prepare_matrix(X))
        class_one = np.flatnonzero(np.asarray(self.estimator.classes_) == 1)
        if len(class_one) != 1:
            raise RuntimeError("Random Forest estimator has no binary positive class labelled 1.")
        return probabilities[:, int(class_one[0])]

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
        importance_type: str = "gini",
        top_k: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract mean impurity-decrease importances as a Polars DataFrame.

        Args:
            importance_type: Accepted for interface symmetry; only 'gini' is available.
            top_k: Optional limit on number of top features to return.

        Returns:
            pl.DataFrame: Table of feature names and relative importances.
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError("Model is not fitted.")

        if importance_type.lower() != "gini":
            raise ValueError("Random Forest supports only importance_type='gini'.")

        raw_importance = np.asarray(self.estimator.feature_importances_, dtype=np.float64)
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
        Serialize the forest, its category dictionaries, and configuration.

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
            "category_maps": self.category_maps_,
            "numeric_medians": self.numeric_medians_,
            "oob_score": self.oob_score_,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Saved Random Forest model to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RandomForestCTRModel":
        """
        Deserialize a saved Random Forest model.

        Args:
            filepath: Path to serialized artifact.

        Returns:
            RandomForestCTRModel: Reconstructed instance.
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
        instance.category_maps_ = payload.get("category_maps", {})
        instance.numeric_medians_ = payload.get("numeric_medians", {})
        instance.oob_score_ = payload.get("oob_score")
        instance.is_fitted = payload.get("is_fitted", False)
        logger.info(f"Loaded Random Forest model from {path}")
        return instance
