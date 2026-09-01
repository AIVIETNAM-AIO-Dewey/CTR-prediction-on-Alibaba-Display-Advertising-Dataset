"""
Abstract Base Class for CTR Prediction Models.
Defines the standard interface for all model wrappers across the project.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
import polars as pl
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class BaseCTRModel(ABC):
    """
    Abstract Base Class for all CTR prediction models.
    Enforces a consistent API for training, inference, evaluation, and persistence.
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.

        Args:
            model_name: Unique identifier for the model architecture.
            config: Dictionary containing model hyperparameters and settings.
        """
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted = False
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(
        self,
        X_train: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
        y_train: Union[pd.Series, pl.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, pl.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, pl.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "BaseCTRModel":
        """
        Train the model on the training partition with optional validation data.

        Args:
            X_train: Feature matrix for training.
            y_train: Binary labels for training (0: non-click, 1: click).
            X_val: Optional feature matrix for validation / early stopping.
            y_val: Optional binary labels for validation.
            **kwargs: Additional model-specific training arguments.

        Returns:
            self: The fitted model instance.
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict click-through probabilities for given features.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: 1D array of predicted click probabilities (p in [0.0, 1.0]).
        """
        pass

    def predict(
        self,
        X: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict discrete binary labels given a decision threshold.

        Args:
            X: Feature matrix.
            threshold: Decision threshold for positive class classification (default: 0.5).

        Returns:
            np.ndarray: 1D array of binary predictions (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.uint8)

    def evaluate(
        self,
        X: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
        y: Union[pd.Series, pl.Series, np.ndarray],
        dataset_name: str = "Test",
    ) -> Dict[str, float]:
        """
        Compute standard CTR prediction benchmark metrics:
        - ROC-AUC: Discrimination ability
        - LogLoss (Binary Cross-Entropy): Probability calibration & loss
        - PR-AUC (Average Precision): Performance under severe class imbalance
        - Brier Score: Mean squared error of calibrated probabilities

        Args:
            X: Feature matrix.
            y: Ground truth binary labels.
            dataset_name: Label for logging (e.g. 'Val', 'Test').

        Returns:
            Dict[str, float]: Computed evaluation metrics.
        """
        if isinstance(y, (pl.Series, pd.Series)):
            y_true = y.to_numpy()
        else:
            y_true = np.asarray(y)

        y_pred_proba = self.predict_proba(X)

        # Clip predicted probabilities to prevent numerical instability in log_loss
        eps = 1e-15
        y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1.0 - eps)

        auc = float(roc_auc_score(y_true, y_pred_proba_clipped))
        loss = float(log_loss(y_true, y_pred_proba_clipped))
        pr_auc = float(average_precision_score(y_true, y_pred_proba_clipped))
        brier = float(brier_score_loss(y_true, y_pred_proba_clipped))

        metrics = {
            f"{dataset_name.lower()}_roc_auc": round(auc, 6),
            f"{dataset_name.lower()}_log_loss": round(loss, 6),
            f"{dataset_name.lower()}_pr_auc": round(pr_auc, 6),
            f"{dataset_name.lower()}_brier_score": round(brier, 6),
        }

        logger.info(
            f"[{self.model_name}] {dataset_name} Evaluation -> "
            f"ROC-AUC: {auc:.5f} | LogLoss: {loss:.5f} | PR-AUC: {pr_auc:.5f} | Brier: {brier:.5f}"
        )
        return metrics

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serialize model and associated metadata to disk.

        Args:
            filepath: Destination file path.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseCTRModel":
        """
        Deserialize model from disk.

        Args:
            filepath: Path to saved model file.

        Returns:
            BaseCTRModel: Loaded instance.
        """
        pass
