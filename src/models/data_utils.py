"""
Data Utilities Module for Model Training and Evaluation.
Provides high-performance loading and preparation of train/val/test splits using Polars.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import polars as pl

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
    "userid",
    "user_id",
]

# Standard categorical feature list in the Alibaba CTR dataset (raw + engineered)
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
    # Engineered Categorical Cross Features
    "gender_x_cate",
    "pid_x_cate",
]

# Standard continuous / numeric feature list (raw + engineered)
DEFAULT_NUMERIC_COLS = [
    "price",
    # Engineered Exposure Sequence Features (Ad fatigue)
    "user_adgroup_exposure_seq",
    "user_cate_exposure_seq",
    # Engineered Price Transforms
    "price_log",
    "price_ratio_cate",
    # Engineered Cyclical Time Encodings
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Out-of-fold Smoothed Bayesian Target Encodings
    "cate_id_te",
    "brand_te",
    "customer_te",
    "pid_te",
]


class CTRDataset:
    """
    Container class holding Polars Train, Validation, and Test feature matrices and labels.
    """

    def __init__(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.Series] = None,
        X_test: Optional[pl.DataFrame] = None,
        y_test: Optional[pl.Series] = None,
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
        """Return dataset partition sizes and baseline CTRs using Polars."""
        train_len = len(self.X_train)
        train_ctr = float(self.y_train.mean() * 100)

        info = {
            "train_samples": train_len,
            "train_ctr": float(train_ctr),
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
    use_fe: bool = False,
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: int = 42,
) -> CTRDataset:
    """
    Load preprocessed or feature-engineered Parquet datasets as native Polars DataFrames and Series.

    Args:
        processed_dir: Directory containing parquet files.
        target_col: Name of the binary target column (default: 'clk').
        exclude_cols: Columns to exclude from features (e.g. timestamps, identifiers).
        categorical_cols: Explicit list of categorical columns.
        numeric_cols: Explicit list of numeric/continuous columns.
        use_fe: If True, loads engineered partitions (train_fe.parquet, etc.).
        sample_size: Optional row limit for rapid baseline testing.
        sample_fraction: Optional fraction for sampling (e.g. 0.1 for 10%).
        random_seed: Seed for sampling reproducibility.

    Returns:
        CTRDataset: Object containing Polars X/y splits and feature metadata.
    """
    proc_path = Path(processed_dir)

    if use_fe:
        train_path = proc_path / "train_fe.parquet"
        val_path = proc_path / "val_fe.parquet"
        test_path = proc_path / "test_fe.parquet"
        if not train_path.exists():
            logger.warning(
                f"Engineered dataset {train_path} not found. Falling back to train.parquet."
            )
            train_path = proc_path / "train.parquet"
            val_path = proc_path / "val.parquet"
            test_path = proc_path / "test.parquet"
    else:
        train_path = proc_path / "train.parquet"
        val_path = proc_path / "val.parquet"
        test_path = proc_path / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {train_path}. Run preprocessing / feature engineering first."
        )

    logger.info(f"Loading datasets from {train_path.parent} (Training file: {train_path.name})...")
    train_pl = pl.read_parquet(train_path)
    val_pl = pl.read_parquet(val_path) if val_path.exists() else None
    test_pl = pl.read_parquet(test_path) if test_path.exists() else None

    # Apply sampling if specified
    if sample_size is not None and sample_size < len(train_pl):
        logger.info(f"Sampling training set to {sample_size:,} rows with Polars...")
        train_pl = train_pl.sample(n=sample_size, seed=random_seed)
        if val_pl is not None:
            val_sample = min(int(sample_size * 0.2), len(val_pl))
            val_pl = val_pl.sample(n=val_sample, seed=random_seed)
        if test_pl is not None:
            test_sample = min(int(sample_size * 0.2), len(test_pl))
            test_pl = test_pl.sample(n=test_sample, seed=random_seed)
    elif sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        logger.info(f"Sampling dataset with fraction {sample_fraction:.2%} with Polars...")
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

    # Safety: dynamically catch any unlisted numerical columns so remainder="drop" doesn't drop them
    for col in feature_cols:
        if col not in active_cats and col not in active_nums:
            col_dtype = train_pl.schema.get(col)
            if col_dtype in (pl.Categorical, pl.Utf8, pl.String, pl.Object):
                active_cats.append(col)
            else:
                active_nums.append(col)

    # Keep native Polars DataFrames and Series
    X_train = train_pl.select(feature_cols)
    y_train = train_pl.select(target_col).to_series()

    X_val = val_pl.select(feature_cols) if val_pl is not None else None
    y_val = val_pl.select(target_col).to_series() if val_pl is not None else None

    X_test = test_pl.select(feature_cols) if test_pl is not None else None
    y_test = test_pl.select(target_col).to_series() if test_pl is not None else None

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
        f"CTRDataset Loaded (Polars) -> Train: {summary['train_samples']:,} rows (CTR: {summary['train_ctr']:.3f}%) | "
        f"Val: {summary.get('val_samples', 0):,} rows | Test: {summary.get('test_samples', 0):,} rows | "
        f"Features: {summary['num_features']} ({summary['categorical_features']} cat, {summary['numeric_features']} num)"
    )

    return dataset
