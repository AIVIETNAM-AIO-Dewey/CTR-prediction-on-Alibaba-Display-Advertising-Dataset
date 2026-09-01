"""
Data Utilities Module for Model Training and Evaluation.
Provides high-performance loading and preparation of train/val/test splits using Polars.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)

# Default column exclusions (targets, timestamps, identifiers)
DEFAULT_TARGET_COL = "clk"
DEFAULT_EXCLUDE_COLS = [
    "clk",
    "nonclk",
    "datetime",
    "date",
    "time_stamp",
    "user",
]

# Standard categorical feature list in the Alibaba CTR dataset
DEFAULT_CATEGORICAL_COLS = [
    "pid",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
    "cms_segid",
    "cms_group_id",
    "cate_id",
    "brand",
    "customer",
    "campaign_id",
    "adgroup_id",
    "is_weekend",
    "day_of_week",
    "hour",
]

# Standard continuous / numeric feature list
DEFAULT_NUMERIC_COLS = [
    "price",
]


class CTRDataset:
    """
    Container class holding Train, Validation, and Test feature matrices and labels.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.categorical_features = categorical_features or []
        self.numeric_features = numeric_features or []

    @property
    def feature_names(self) -> List[str]:
        return list(self.X_train.columns)

    def summary(self) -> Dict[str, Any]:
        """Return dataset partition sizes and baseline CTRs."""
        info = {
            "train_samples": len(self.X_train),
            "train_ctr": float(self.y_train.mean() * 100),
            "num_features": len(self.feature_names),
            "categorical_features": len(self.categorical_features),
            "numeric_features": len(self.numeric_features),
        }
        if self.X_val is not None and self.y_val is not None:
            info["val_samples"] = len(self.X_val)
            info["val_ctr"] = float(self.y_val.mean() * 100)
        if self.X_test is not None and self.y_test is not None:
            info["test_samples"] = len(self.X_test)
            info["test_ctr"] = float(self.y_test.mean() * 100)
        return info


def load_ctr_dataset(
    processed_dir: Union[str, Path] = "data/processed",
    target_col: str = DEFAULT_TARGET_COL,
    exclude_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: int = 42,
) -> CTRDataset:
    """
    Load preprocessed Parquet datasets and split into X and y for Train, Val, and Test.

    Args:
        processed_dir: Directory containing train.parquet, val.parquet, and test.parquet.
        target_col: Name of the binary target column (default: 'clk').
        exclude_cols: Columns to exclude from features (e.g. timestamps, identifiers).
        categorical_cols: Explicit list of categorical columns.
        numeric_cols: Explicit list of numeric/continuous columns.
        sample_size: Optional row limit for rapid baseline testing.
        sample_fraction: Optional fraction for sampling (e.g. 0.1 for 10%).
        random_seed: Seed for sampling reproducibility.

    Returns:
        CTRDataset: Object containing X/y splits and feature metadata.
    """
    proc_path = Path(processed_dir)
    train_path = proc_path / "train.parquet"
    val_path = proc_path / "val.parquet"
    test_path = proc_path / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_path}. Run preprocessing first.")

    logger.info(f"Loading datasets from {proc_path}...")
    train_pl = pl.read_parquet(train_path)
    val_pl = pl.read_parquet(val_path) if val_path.exists() else None
    test_pl = pl.read_parquet(test_path) if test_path.exists() else None

    # Apply sampling if specified
    if sample_size is not None and sample_size < len(train_pl):
        logger.info(f"Sampling training set to {sample_size:,} rows...")
        train_pl = train_pl.sample(n=sample_size, seed=random_seed)
        if val_pl is not None:
            val_sample = min(int(sample_size * 0.2), len(val_pl))
            val_pl = val_pl.sample(n=val_sample, seed=random_seed)
        if test_pl is not None:
            test_sample = min(int(sample_size * 0.2), len(test_pl))
            test_pl = test_pl.sample(n=test_sample, seed=random_seed)
    elif sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        logger.info(f"Sampling dataset with fraction {sample_fraction:.2%}...")
        train_pl = train_pl.sample(fraction=sample_fraction, seed=random_seed)
        if val_pl is not None:
            val_pl = val_pl.sample(fraction=sample_fraction, seed=random_seed)
        if test_pl is not None:
            test_pl = test_pl.sample(fraction=sample_fraction, seed=random_seed)

    # Determine feature column names
    excluded = set(exclude_cols or DEFAULT_EXCLUDE_COLS)
    all_cols = train_pl.columns
    feature_cols = [c for c in all_cols if c not in excluded and c != target_col]

    # Classify feature types
    cat_candidates = categorical_cols or DEFAULT_CATEGORICAL_COLS
    num_candidates = numeric_cols or DEFAULT_NUMERIC_COLS

    active_cats = [c for c in cat_candidates if c in feature_cols]
    active_nums = [c for c in num_candidates if c in feature_cols]

    # Convert to Pandas for model compatibility
    train_pd = train_pl.select(feature_cols + [target_col]).to_pandas()
    X_train = train_pd[feature_cols]
    y_train = train_pd[target_col]

    X_val, y_val = None, None
    if val_pl is not None:
        val_pd = val_pl.select(feature_cols + [target_col]).to_pandas()
        X_val = val_pd[feature_cols]
        y_val = val_pd[target_col]

    X_test, y_test = None, None
    if test_pl is not None:
        test_pd = test_pl.select(feature_cols + [target_col]).to_pandas()
        X_test = test_pd[feature_cols]
        y_test = test_pd[target_col]

    dataset = CTRDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_features=active_cats,
        numeric_features=active_nums,
    )

    summary = dataset.summary()
    logger.info(
        f"CTRDataset Loaded -> Train: {summary['train_samples']:,} rows (CTR: {summary['train_ctr']:.3f}%) | "
        f"Val: {summary.get('val_samples', 0):,} rows | Test: {summary.get('test_samples', 0):,} rows | "
        f"Features: {summary['num_features']} ({summary['categorical_features']} cat, {summary['numeric_features']} num)"
    )

    return dataset
