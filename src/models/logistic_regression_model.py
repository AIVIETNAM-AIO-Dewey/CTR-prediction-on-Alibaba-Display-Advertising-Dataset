"""
Logistic Regression Baseline Model for CTR Prediction.
Implements a standardized linear baseline with robust sparse categorical encoding,
numeric scaling, and probability calibration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.base_model import BaseCTRModel

logger = logging.getLogger(__name__)


class LogisticRegressionCTRModel(BaseCTRModel):
    """
    Logistic Regression Model Wrapper for Click-Through Rate (CTR) Prediction.
    Features:
    - Automated sparse One-Hot Encoding for categorical features with min_frequency / max_categories.
    - StandardScaler for numeric features.
    - End-to-end scikit-learn Pipeline with serialization.
    - Support for standard L-BFGS or scalable SGD (log-loss) for massive dataset partitions.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 300,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        use_sgd: bool = False,
        sgd_alpha: float = 1e-4,
        max_categories_per_feature: Optional[int] = 200,
        min_category_frequency: Optional[int] = 10,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        random_state: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LogisticRegressionCTRModel.

        Args:
            C: Inverse regularization strength.
            penalty: Regularization norm ('l1', 'l2', 'elasticnet').
            solver: Optimization solver ('lbfgs', 'saga', 'liblinear').
            max_iter: Maximum optimization iterations.
            class_weight: Optional weights associated with classes ('balanced' or dict).
            use_sgd: If True, uses SGDClassifier(loss='log_loss') for high memory scalability.
            sgd_alpha: Regularization multiplier for SGD solver.
            max_categories_per_feature: Maximum one-hot categories per feature (infrequent grouped to 'infrequent_sklearn').
            min_category_frequency: Minimum occurrences to qualify for dedicated one-hot bin.
            categorical_features: Explicit list of categorical column names.
            numeric_features: Explicit list of numeric column names.
            random_state: Seed for reproducibility.
            config: Additional hyperparameter dictionary.
        """
        super().__init__(model_name="LogisticRegression", config=config)

        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.use_sgd = use_sgd
        self.sgd_alpha = sgd_alpha
        self.max_categories = max_categories_per_feature
        self.min_frequency = min_category_frequency
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.random_state = random_state

        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.estimator: Optional[Union[LogisticRegression, SGDClassifier]] = None

    def _build_pipeline(
        self,
        categorical_cols: List[str],
        numeric_cols: List[str],
    ) -> Pipeline:
        """
        Construct ColumnTransformer and model Pipeline.

        Args:
            categorical_cols: Names of categorical columns.
            numeric_cols: Names of numeric columns.

        Returns:
            Pipeline: Scikit-learn Pipeline instance.
        """
        transformers = []

        if categorical_cols:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                max_categories=self.max_categories,
                min_frequency=self.min_frequency,
            )
            transformers.append(("cat", ohe, categorical_cols))

        if numeric_cols:
            scaler = StandardScaler(with_mean=False)
            transformers.append(("num", scaler, numeric_cols))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=1.0,
        )

        if self.use_sgd:
            self.estimator = SGDClassifier(
                loss="log_loss",
                penalty=self.penalty,
                alpha=self.sgd_alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
        else:
            self.estimator = LogisticRegression(
                C=self.C,
                penalty=self.penalty,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=-1 if self.solver in ("lbfgs", "saga") else 1,
            )

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", self.estimator),
        ])

        return pipeline

    def fit(
        self,
        X_train: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
        y_train: Union[pd.Series, pl.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, pl.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "LogisticRegressionCTRModel":
        """
        Fit Logistic Regression pipeline on the training set.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels (0 or 1).
            X_val: Optional validation features (evaluated after training).
            y_val: Optional validation labels.
            **kwargs: Additional arguments.

        Returns:
            self: The fitted model.
        """
        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.to_pandas()
        if isinstance(y_train, pl.Series):
            y_train = y_train.to_numpy()

        # Determine column lists
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            cat_cols = self.categorical_features or [
                c for c in self.feature_names if c != "price"
            ]
            num_cols = self.numeric_features or [
                c for c in self.feature_names if c == "price"
            ]
            # Ensure categorical types are cast to object/string for OHE stability
            X_train_processed = X_train.copy()
            for col in cat_cols:
                if col in X_train_processed.columns:
                    X_train_processed[col] = X_train_processed[col].astype(str)
        else:
            raise TypeError("X_train must be a pandas DataFrame or Polars DataFrame.")

        logger.info(f"Building Logistic Regression pipeline ({len(cat_cols)} cat, {len(num_cols)} num features)...")
        self.pipeline = self._build_pipeline(cat_cols, num_cols)

        logger.info(f"Fitting Logistic Regression model on {len(X_train_processed):,} training samples...")
        self.pipeline.fit(X_train_processed, y_train)
        self.is_fitted = True
        logger.info("✅ Logistic Regression model training completed successfully.")

        # If validation data is provided, evaluate and log
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
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() before predicting.")

        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()

        if isinstance(X, pd.DataFrame):
            X_eval = X.copy()
            # Cast categorical columns to string matching training
            cat_cols = self.categorical_features or [
                c for c in X_eval.columns if c != "price"
            ]
            for col in cat_cols:
                if col in X_eval.columns:
                    X_eval[col] = X_eval[col].astype(str)
        else:
            X_eval = X

        # Predict probability for positive class (class 1: click)
        probas = self.pipeline.predict_proba(X_eval)
        return probas[:, 1]

    def get_top_coefficients(self, top_k: int = 20) -> pd.DataFrame:
        """
        Extract top positive and negative feature coefficients from the fitted linear model.

        Args:
            top_k: Number of top influential features to return.

        Returns:
            pd.DataFrame: Table of feature names and linear weights.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted.")

        preprocessor = self.pipeline.named_steps["preprocessor"]
        classifier = self.pipeline.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        df_coef = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_importance": np.abs(coefficients),
        }).sort_values(by="abs_importance", ascending=False)

        return df_coef.head(top_k)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serialize model pipeline and metadata using Joblib.

        Args:
            filepath: Destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "pipeline": self.pipeline,
            "config": self.config,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Saved Logistic Regression model to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LogisticRegressionCTRModel":
        """
        Deserialize saved model pipeline.

        Args:
            filepath: File path to serialized artifact.

        Returns:
            LogisticRegressionCTRModel: Reconstructed instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        payload = joblib.load(path)
        instance = cls(
            categorical_features=payload.get("categorical_features"),
            numeric_features=payload.get("numeric_features"),
            config=payload.get("config"),
        )
        instance.pipeline = payload.get("pipeline")
        instance.feature_names = payload.get("feature_names", [])
        instance.is_fitted = payload.get("is_fitted", False)
        logger.info(f"Loaded Logistic Regression model from {path}")
        return instance
