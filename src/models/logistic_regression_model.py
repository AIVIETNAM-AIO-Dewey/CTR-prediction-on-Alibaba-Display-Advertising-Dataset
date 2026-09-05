"""
Logistic Regression Model for CTR Prediction.
Self-contained linear baseline with native Polars preprocessing,
sparse one-hot encoding for categoricals, feature scaling for numerics,
and support for both standard LogisticRegression (L-BFGS) and SGDClassifier (log loss).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

logger = logging.getLogger(__name__)




class LogisticRegressionModel:
    """
    Logistic Regression Model for Click-Through Rate (CTR) Prediction.

    Key Capabilities:
    - Pure Polars input handling with zero pandas conversion.
    - Sparse One-Hot Encoding for categorical features.
    - StandardScaler with mean centering disabled to preserve sparse matrices.
    - Supports exact L-BFGS optimization (`use_sgd=False`) or scalable SGD (`use_sgd=True`).
    - Built-in evaluate(X, y) calculating LogLoss and ROC-AUC.
    - End-to-end execution directly from config: `LogisticRegressionModel.fit_from_config(...)`.
    - Exposes top positive and negative coefficient feature diagnostics as Polars DataFrames.
    - Serialization (save / load) and YAML configuration loader (from_config).
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 200,
        use_sgd: bool = False,
        alpha: float = 1e-4,
        class_weight: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        **extra_kwargs: Any,
    ):
        self.model_name = "LogisticRegression"
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.use_sgd = use_sgd or (solver == "sgd")
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.categorical_features = list(categorical_features or [])
        self.numeric_features = list(numeric_features or [])
        self.config = config or {}
        self.extra_kwargs = extra_kwargs

        self.feature_names: List[str] = []
        self.scaler_: Optional[StandardScaler] = None
        self.ohe_: Optional[OneHotEncoder] = None
        self.estimator: Optional[Union[LogisticRegression, SGDClassifier]] = None
        self.transformed_feature_names_: List[str] = []
        self.best_iteration_: int = 0
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
    ) -> "LogisticRegressionModel":
        """Instantiate LogisticRegressionModel directly from a YAML config with optional keyword overrides."""
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
        config_path_or_dict: Union[str, Path, Dict[str, Any]] = "configs/logistic_regression.yaml",
        sample_size: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        data_dir: Optional[Union[str, Path]] = None,
        save_artifact: bool = True,
        models_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Tuple["LogisticRegressionModel", Dict[str, float]]:
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
            basename = paths_cfg.get("model_basename", "logistic_regression")
            use_fe = cfg.get("data", {}).get("use_fe", True)
            save_path = out_dir / f"{basename}{'_fe' if use_fe else '_baseline'}.joblib"
            model.save(save_path)

        return model, metrics

    def _split_feature_cols(self, df: pl.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify active categorical and numeric columns present in the dataframe."""
        all_cols = set(df.columns)
        cat_cols = [c for c in self.categorical_features if c in all_cols]
        num_cols = [c for c in self.numeric_features if c in all_cols]

        for col in df.columns:
            if col not in cat_cols and col not in num_cols:
                dtype = df.schema.get(col)
                if dtype in (pl.Categorical, pl.Utf8, pl.String, pl.Object):
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
        return cat_cols, num_cols

    def _transform_features(
        self,
        df: Union[pl.DataFrame, np.ndarray],
        fit_transformers: bool = False,
    ) -> sp.csr_matrix:
        """
        Process Polars DataFrame into a sparse feature matrix.
        - Numeric features: imputed with 0.0 and scaled with StandardScaler(with_mean=False).
        - Categorical features: cast to String, nulls filled, and encoded with OneHotEncoder.
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        elif isinstance(df, np.ndarray):
            df = pl.DataFrame(df, schema=self.feature_names)
        elif not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, Pandas DataFrame, or numpy ndarray, got {type(df)}")

        cat_cols, num_cols = self._split_feature_cols(df)

        parts: List[sp.spmatrix] = []
        out_names: List[str] = []

        if num_cols:
            num_exprs = [pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in num_cols]
            X_num_np = df.select(num_exprs).to_numpy()

            if fit_transformers:
                self.scaler_ = StandardScaler(with_mean=False)
                X_num_scaled = self.scaler_.fit_transform(X_num_np)
            else:
                if self.scaler_ is None:
                    raise RuntimeError("StandardScaler was not fitted.")
                X_num_scaled = self.scaler_.transform(X_num_np)

            parts.append(sp.csr_matrix(X_num_scaled))
            out_names.extend(num_cols)

        if cat_cols:
            cat_exprs = [pl.col(c).cast(pl.String).fill_null("__NA__").alias(c) for c in cat_cols]
            X_cat_np = df.select(cat_exprs).to_numpy()

            if fit_transformers:
                try:
                    self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                except TypeError:
                    self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse=True)
                X_cat_sparse = self.ohe_.fit_transform(X_cat_np)
            else:
                if self.ohe_ is None:
                    raise RuntimeError("OneHotEncoder was not fitted.")
                X_cat_sparse = self.ohe_.transform(X_cat_np)

            parts.append(X_cat_sparse)
            try:
                cat_feature_names = list(self.ohe_.get_feature_names_out(cat_cols))
            except Exception:
                cat_feature_names = [f"cat_{i}" for i in range(X_cat_sparse.shape[1])]
            out_names.extend(cat_feature_names)

        if fit_transformers:
            self.transformed_feature_names_ = out_names

        if not parts:
            raise ValueError("No features available to transform.")

        if len(parts) == 1:
            return parts[0].tocsr()
        return sp.hstack(parts, format="csr")

    def fit(
        self,
        X: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pl.Series, np.ndarray]] = None,
        X_train: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_train: Optional[Union[pl.Series, np.ndarray]] = None,
        X_val: Optional[Union[pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pl.Series, np.ndarray]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any,
    ) -> "LogisticRegressionModel":
        """Fit Logistic Regression model using Polars features."""
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

        y_tr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y).ravel()

        logger.info(
            f"[{self.model_name}] Preprocessing {len(self.feature_names)} features (Polars -> CSR)..."
        )
        X_tr_csr = self._transform_features(X, fit_transformers=True)

        if self.use_sgd:
            logger.info(
                f"[{self.model_name}] Training SGDClassifier(loss='log_loss', alpha={self.alpha}, max_iter={self.max_iter}) "
                f"on {X_tr_csr.shape[0]:,} samples x {X_tr_csr.shape[1]:,} sparse features."
            )
            try:
                self.estimator = SGDClassifier(
                    loss="log_loss",
                    penalty=self.penalty,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                )
            except ValueError:
                self.estimator = SGDClassifier(
                    loss="log",
                    penalty=self.penalty,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                )
        else:
            logger.info(
                f"[{self.model_name}] Training LogisticRegression(solver='{self.solver}', C={self.C}, max_iter={self.max_iter}) "
                f"on {X_tr_csr.shape[0]:,} samples x {X_tr_csr.shape[1]:,} sparse features."
            )
            self.estimator = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        self.estimator.fit(X_tr_csr, y_tr)
        self.is_fitted = True
        self.best_iteration_ = int(
            getattr(self.estimator, "n_iter_", [0])[0]
            if hasattr(getattr(self.estimator, "n_iter_", 0), "__getitem__")
            else getattr(self.estimator, "n_iter_", 0)
        )

        logger.info(
            f"[{self.model_name}] Training complete in {self.best_iteration_} iterations."
        )
        return self

    def predict_proba(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict positive click probabilities (1D float array in [0.0, 1.0])."""
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted before predicting.")

        X_csr = self._transform_features(X, fit_transformers=False)

        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X_csr)
            return proba[:, 1].astype(np.float64)
        else:
            decision = self.estimator.decision_function(X_csr)
            return (1.0 / (1.0 + np.exp(-decision))).astype(np.float64)

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

    def get_feature_importance(self, top_k: Optional[int] = None) -> pl.DataFrame:
        """Return the top model coefficients sorted by absolute magnitude as a Polars DataFrame."""
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"[{self.model_name}] Model must be fitted first.")

        coefs = self.estimator.coef_.ravel()
        names = self.transformed_feature_names_ or [f"feat_{i}" for i in range(len(coefs))]

        df_imp = pl.DataFrame({
            "feature": names,
            "coefficient": coefs,
            "abs_importance": np.abs(coefs),
        }).sort("abs_importance", descending=True)

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
    def load(cls, filepath: Union[str, Path]) -> "LogisticRegressionModel":
        """Load serialized model artifact from disk."""
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at: {load_path}")
        instance = joblib.load(load_path)
        logger.info(f"[{instance.model_name}] Model successfully loaded from {load_path}")
        return instance


# Alias for backward compatibility
LogisticRegressionCTRModel = LogisticRegressionModel
