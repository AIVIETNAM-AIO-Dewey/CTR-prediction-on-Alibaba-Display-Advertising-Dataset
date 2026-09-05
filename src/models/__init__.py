"""
Models Package for CTR Prediction.
Provides standardized model wrappers, evaluation interfaces, and dataset loaders.
"""

from src.models.base_model import BaseCTRModel
from src.models.data_utils import CTRDataset, load_ctr_dataset
from src.models.catboost_model import CatBoostCTRModel
from src.models.xgboost_model import XGBoostCTRModel

__all__ = [
    "BaseCTRModel",
    "CTRDataset",
    "load_ctr_dataset",
    "CatBoostCTRModel",
    "XGBoostCTRModel",
]
