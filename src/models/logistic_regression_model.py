"""
Logistic Regression Baseline Model for CTR Prediction.
Implements a standardized linear baseline with robust sparse categorical encoding,
numeric scaling, probability calibration, and native Polars DataFrame processing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import joblib
import numpy as np
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
    - Native Polars DataFrame processing.
    - Automated sparse One-Hot Encoding for categorical features with min_frequency / max_categories.
    - StandardScaler for numeric features.
    - End-to-end scikit-learn Pipeline with serialization.
    - Top coefficients extraction returning Polars DataFrames.
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
            # On massive datasets (20M+ rows), 5-15 epochs with early stopping is optimal
            sgd_epochs = min(self.max_iter, 15)
            self.estimator = SGDClassifier(
                loss="log_loss",
                penalty=self.penalty,
                alpha=self.sgd_alpha,
                max_iter=sgd_epochs,
                tol=1e-3,
                early_stopping=True,
                n_iter_no_change=3,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
        else:
            lr_kwargs: Dict[str, Any] = {
                "C": self.C,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                "class_weight": self.class_weight,
            }
            if self.penalty and self.penalty != "l2":
                lr_kwargs["penalty"] = self.penalty
            self.estimator = LogisticRegression(**lr_kwargs)

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", self.estimator),
        ])

        return pipeline

    def _prepare_dataframe(
        self,
        df: Union[pl.DataFrame, np.ndarray],
        cat_cols: List[str],
    ) -> pl.DataFrame:
        """
        Format Polars DataFrame for Logistic Regression ColumnTransformer.
        Ensures categorical columns are cast to string in Polars for robust OneHotEncoding.

        Args:
            df: Input feature matrix.
            cat_cols: List of categorical column names.

        Returns:
            pl.DataFrame: Formatted Polars DataFrame.
        """
        if isinstance(df, np.ndarray):
            df = pl.DataFrame(df, schema=self.feature_names)
        elif not isinstance(df, pl.DataFrame):
            raise TypeError("Expected Polars DataFrame or numpy ndarray.")

        exprs = [
            pl.col(c).cast(pl.String).alias(c)
            for c in cat_cols
            if c in df.columns
        ]
        if exprs:
            df = df.with_columns(exprs)
        return df

    def fit(
        self,
        X_train: Union[pl.DataFrame, np.ndarray],
        y_train: Union[pl.Series, np.ndarray],
        X_val: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pl.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "LogisticRegressionCTRModel":
        """
        Fit Logistic Regression pipeline on the training set.

        Args:
            X_train: Training feature matrix (Polars DataFrame preferred).
            y_train: Training labels (0 or 1).
            X_val: Optional validation features (evaluated after training).
            y_val: Optional validation labels.
            **kwargs: Additional arguments.

        Returns:
            self: The fitted model.
        """
        if isinstance(y_train, pl.Series):
            y_train = y_train.to_numpy()

        if isinstance(X_train, pl.DataFrame):
            self.feature_names = list(X_train.columns)
            cat_cols = self.categorical_features or [
                c for c in self.feature_names if c != "price"
            ]
            num_cols = self.numeric_features or [
                c for c in self.feature_names if c == "price"
            ]
        elif isinstance(X_train, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            cat_cols = self.categorical_features or []
            num_cols = self.numeric_features or self.feature_names
        else:
            raise TypeError("X_train must be a Polars DataFrame or numpy ndarray.")

        active_cat_cols = [c for c in cat_cols if c in self.feature_names]
        active_num_cols = [c for c in num_cols if c in self.feature_names]

        X_train_processed = self._prepare_dataframe(X_train, active_cat_cols)

        logger.info(
            f"Building Logistic Regression pipeline (Polars) ({len(active_cat_cols)} cat, {len(active_num_cols)} num features)..."
        )
        self.pipeline = self._build_pipeline(active_cat_cols, active_num_cols)

        logger.info(f"Fitting Logistic Regression model on {len(X_train_processed):,} training samples...")
        self.pipeline.fit(X_train_processed, y_train)
        self.is_fitted = True
        logger.info("✅ Logistic Regression model training completed successfully.")

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
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() before predicting.")

        if isinstance(X, pl.DataFrame):
            cat_cols = self.categorical_features or [
                c for c in X.columns if c != "price"
            ]
            X_eval = self._prepare_dataframe(X, cat_cols)
        elif isinstance(X, np.ndarray):
            cat_cols = self.categorical_features or []
            X_eval = self._prepare_dataframe(X, cat_cols)
        else:
            raise TypeError("X must be a Polars DataFrame or numpy ndarray.")

        probas = self.pipeline.predict_proba(X_eval)
        return probas[:, 1]

    def get_top_coefficients(self, top_k: int = 20) -> pl.DataFrame:
        """
        Extract top positive and negative feature coefficients from the fitted linear model as a Polars DataFrame.

        Args:
            top_k: Number of top influential features to return.

        Returns:
            pl.DataFrame: Table of feature names and linear weights.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted.")

        preprocessor = self.pipeline.named_steps["preprocessor"]
        classifier = self.pipeline.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        df_coef = pl.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients.astype(np.float64),
            "abs_importance": np.abs(coefficients).astype(np.float64),
        }).sort(by="abs_importance", descending=True)

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
