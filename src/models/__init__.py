"""
Models Package for CTR Prediction on Alibaba Display Advertising Dataset.

Provides model wrappers and training pipelines for:
- CatBoost (CatBoostCTRModel)
- XGBoost (XGBoostCTRModel)
- LightGBM (LightGBMModel)
- Logistic Regression (LogisticRegressionModel)
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.models.catboost_model import CatBoostCTRModel
    from src.models.xgboost_model import XGBoostCTRModel
    from src.models.lightgbm_model import LightGBMModel
    from src.models.logistic_regression_model import LogisticRegressionModel
    from src.models.train import (
        CTRDataset,
        FitResult,
        MODEL_REGISTRY,
        fit_from_config,
        load_ctr_dataset,
        load_dataset_from_config,
    )

    LightGBMCTRModel = LightGBMModel
    LogisticRegressionCTRModel = LogisticRegressionModel


def __getattr__(name: str) -> Any:
    """Lazy-load models and utilities on demand to prevent missing-library crashes."""
    if name == "CatBoostCTRModel":
        from src.models.catboost_model import CatBoostCTRModel
        return CatBoostCTRModel
    if name == "XGBoostCTRModel":
        from src.models.xgboost_model import XGBoostCTRModel
        return XGBoostCTRModel
    if name in ("LightGBMModel", "LightGBMCTRModel"):
        from src.models.lightgbm_model import LightGBMModel
        return LightGBMModel
    if name in ("LogisticRegressionModel", "LogisticRegressionCTRModel"):
        from src.models.logistic_regression_model import LogisticRegressionModel
        return LogisticRegressionModel
    if name in (
        "CTRDataset",
        "FitResult",
        "MODEL_REGISTRY",
        "fit_from_config",
        "load_ctr_dataset",
        "load_dataset_from_config",
    ):
        import src.models.train as _train
        return getattr(_train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CatBoostCTRModel",
    "XGBoostCTRModel",
    "LightGBMModel",
    "LightGBMCTRModel",
    "LogisticRegressionModel",
    "LogisticRegressionCTRModel",
    "CTRDataset",
    "FitResult",
    "MODEL_REGISTRY",
    "fit_from_config",
    "load_ctr_dataset",
    "load_dataset_from_config",
]
