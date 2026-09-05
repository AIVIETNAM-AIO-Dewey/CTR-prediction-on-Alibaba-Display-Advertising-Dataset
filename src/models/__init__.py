"""Lazy public exports for the tree-model wrappers.

The shared loader and training orchestration are intentionally kept out of this module. This
lets callers import one wrapper without eagerly importing every optional model backend.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Load a model wrapper only when it is requested."""
    if name == "CatBoostCTRModel":
        from src.models.catboost_model import CatBoostCTRModel

        return CatBoostCTRModel
    if name == "XGBoostCTRModel":
        from src.models.xgboost_model import XGBoostCTRModel

        return XGBoostCTRModel
    if name == "RandomForestCTRModel":
        from src.models.random_forest_model import RandomForestCTRModel

        return RandomForestCTRModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CatBoostCTRModel", "XGBoostCTRModel", "RandomForestCTRModel"]
