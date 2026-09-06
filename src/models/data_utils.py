"""Shared parquet loading for fit-only model entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import polars as pl


@dataclass
class CTRDataset:
    """Train/validation/test partitions with one frozen feature schema."""

    X_train: pl.DataFrame
    y_train: pl.Series
    X_val: Optional[pl.DataFrame] = None
    y_val: Optional[pl.Series] = None
    X_test: Optional[pl.DataFrame] = None
    y_test: Optional[pl.Series] = None
    categorical_features: List[str] = field(default_factory=list)
    numeric_features: List[str] = field(default_factory=list)

    @property
    def feature_names(self) -> List[str]:
        return list(self.X_train.columns)


def _sample(
    frame: pl.DataFrame,
    sample_size: Optional[int],
    sample_fraction: Optional[float],
    seed: int,
) -> pl.DataFrame:
    if sample_size is not None:
        return frame.sample(n=min(sample_size, len(frame)), shuffle=True, seed=seed)
    if sample_fraction is not None and sample_fraction < 1:
        return frame.sample(fraction=sample_fraction, shuffle=True, seed=seed)
    return frame


def _load_optional(path: Path) -> Optional[pl.DataFrame]:
    return pl.read_parquet(path) if path.exists() else None


def load_ctr_dataset(
    processed_dir: str = "data/processed",
    target_col: str = "clk",
    exclude_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    numeric_cols: Optional[Sequence[str]] = None,
    use_fe: bool = True,
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_seed: int = 42,
) -> CTRDataset:
    """Load parquet partitions and freeze their feature order to the train schema."""
    if sample_size is not None and sample_size <= 0:
        sample_size = None
    if sample_fraction is not None and not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    if sample_size is not None and sample_fraction is not None:
        raise ValueError("Use either sample_size or sample_fraction, not both.")

    directory = Path(processed_dir)
    suffix = "_fe" if use_fe else ""
    train_path = directory / f"train{suffix}.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training partition not found: {train_path}")

    train = pl.read_parquet(train_path)
    validation = _load_optional(directory / f"val{suffix}.parquet")
    test = _load_optional(directory / f"test{suffix}.parquet")
    partitions = {"train": train, "validation": validation, "test": test}
    for name, frame in partitions.items():
        if frame is not None and target_col not in frame.columns:
            raise ValueError(f"Target column '{target_col}' is missing from the {name} partition.")

    excluded = set(exclude_cols or []) | {target_col}
    features = [column for column in train.columns if column not in excluded]
    if not features:
        raise ValueError("No model features remain after applying exclude_cols.")

    for name, frame in partitions.items():
        if frame is None:
            continue
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise ValueError(f"{name.capitalize()} partition is missing feature(s): {missing}")

    configured_cats = [column for column in (categorical_cols or []) if column in features]
    configured_nums = [
        column
        for column in (numeric_cols or [])
        if column in features and column not in configured_cats
    ]
    cats = list(configured_cats)
    nums = list(configured_nums)
    for column in features:
        if column in cats or column in nums:
            continue
        dtype = train.schema[column]
        if dtype in (pl.String, pl.Categorical, pl.Enum, pl.Object):
            cats.append(column)
        else:
            nums.append(column)

    sampled = {
        name: _sample(frame, sample_size, sample_fraction, random_seed)
        if frame is not None
        else None
        for name, frame in partitions.items()
    }

    def split(frame: Optional[pl.DataFrame]):
        if frame is None:
            return None, None
        return frame.select(features), frame.get_column(target_col)

    X_train, y_train = split(sampled["train"])
    X_val, y_val = split(sampled["validation"])
    X_test, y_test = split(sampled["test"])
    assert X_train is not None and y_train is not None
    return CTRDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_features=cats,
        numeric_features=nums,
    )
