"""
Feature Engineering Module for CTR Prediction.

Implements exposure-sequence counters, price transformations, cyclical time
encodings, cross features, and out-of-fold smoothed Bayesian target encoding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import math

import numpy as np
import polars as pl
import yaml

logger = logging.getLogger(__name__)

_SPLIT_COL = "_split"


class CTRFeatureEngineer:
    """Generates model-ready features for the Alibaba CTR dataset."""

    def __init__(
        self,
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        target_encode_cols: Optional[List[str]] = None,
        smoothing: float = 20.0,
        n_folds: int = 5,
        random_seed: int = 42,
    ):
        cfg = self._load_config(config)
        te_cfg = cfg.get("feature_engineering", {}).get("target_encoding", {})

        self.target_encode_cols = (
            target_encode_cols
            or te_cfg.get("columns")
            or ["cate_id", "brand", "customer", "pid"]
        )
        self.smoothing = te_cfg.get("smoothing", smoothing)
        self.n_folds = te_cfg.get("n_folds", n_folds)
        self.random_seed = te_cfg.get("random_seed", random_seed)

        self.cate_median_price: Optional[pl.DataFrame] = None
        self.global_median_price: Optional[float] = None
        self.global_ctr: Optional[float] = None
        self.target_encoding_maps: Dict[str, pl.DataFrame] = {}

    @staticmethod
    def _load_config(config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        """Load YAML config or return dictionary."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        config_path = Path(config)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"Config file not found at {config_path}, using defaults.")
        return {}

    # ------------------------------------------------------------------ #
    # Exposure Sequence (Ad Fatigue)
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_exposure_sequence(df: pl.DataFrame) -> pl.DataFrame:
        """Add prior-exposure counts per (user, adgroup_id) and (user, cate_id)."""
        logger.info("Computing exposure sequence counters (ad fatigue)...")
        df = df.sort("time_stamp")
        df = df.with_columns([
            (pl.col("time_stamp").cum_count().over(["user", "adgroup_id"]) - 1)
            .cast(pl.Int32)
            .alias("user_adgroup_exposure_seq"),
            (pl.col("time_stamp").cum_count().over(["user", "cate_id"]) - 1)
            .cast(pl.Int32)
            .alias("user_cate_exposure_seq"),
        ])
        return df

    # ------------------------------------------------------------------ #
    # Price Transformations
    # ------------------------------------------------------------------ #
    def fit_price_stats(self, train_df: pl.DataFrame) -> "CTRFeatureEngineer":
        """Fit per-category median price on the training partition."""
        logger.info("Fitting per-category median price statistics on train partition...")
        self.cate_median_price = (
            train_df.group_by("cate_id")
            .agg(pl.col("price").median().alias("cate_median_price"))
        )
        self.global_median_price = float(train_df.select(pl.col("price").median()).item() or 0.0)
        return self

    def add_price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add `price_log` (log1p) and `price_ratio_cate` (price / train-fitted category median)."""
        if self.cate_median_price is None:
            raise RuntimeError("fit_price_stats() must be called before add_price_features().")

        logger.info("Adding price_log and price_ratio_cate features...")
        fallback = self.global_median_price or 1.0

        df = df.join(self.cate_median_price, on="cate_id", how="left")
        df = df.with_columns([
            pl.col("price").log1p().cast(pl.Float32).alias("price_log"),
            pl.col("cate_median_price").fill_null(fallback).alias("cate_median_price"),
        ])
        df = df.with_columns([
            (
                pl.col("price")
                / pl.when(pl.col("cate_median_price") > 0)
                .then(pl.col("cate_median_price"))
                .otherwise(fallback)
            )
            .cast(pl.Float32)
            .alias("price_ratio_cate")
        ]).drop("cate_median_price")
        return df

    # ------------------------------------------------------------------ #
    # Cyclical Time Encodings
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_cyclical_time_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add sine/cosine encodings for `hour` and `day_of_week`."""
        logger.info("Adding cyclical time encodings for hour and day_of_week...")
        two_pi = 2.0 * math.pi
        df = df.with_columns([
            (pl.col("hour").cast(pl.Float64) * (two_pi / 24)).sin().cast(pl.Float32).alias("hour_sin"),
            (pl.col("hour").cast(pl.Float64) * (two_pi / 24)).cos().cast(pl.Float32).alias("hour_cos"),
            (pl.col("day_of_week").cast(pl.Float64) * (two_pi / 7)).sin().cast(pl.Float32).alias("dow_sin"),
            (pl.col("day_of_week").cast(pl.Float64) * (two_pi / 7)).cos().cast(pl.Float32).alias("dow_cos"),
        ])
        return df

    # ------------------------------------------------------------------ #
    # Cross Features
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_cross_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add `gender_x_cate` and `pid_x_cate` categorical cross features."""
        logger.info("Adding cross features (final_gender_code x cate_id, pid x cate_id)...")
        df = df.with_columns([
            pl.concat_str(
                [pl.col("final_gender_code").cast(pl.Utf8), pl.col("cate_id").cast(pl.Utf8)],
                separator="_",
            ).cast(pl.Categorical).alias("gender_x_cate"),
            pl.concat_str(
                [pl.col("pid").cast(pl.Utf8), pl.col("cate_id").cast(pl.Utf8)],
                separator="_",
            ).cast(pl.Categorical).alias("pid_x_cate"),
        ])
        return df

    # ------------------------------------------------------------------ #
    # Out-of-Fold Smoothed Bayesian Target Encoding
    # ------------------------------------------------------------------ #
    def _fit_te_lookup(self, fit_df: pl.DataFrame, col: str, prior: float) -> pl.DataFrame:
        """Build a {col, col_te} smoothed target-encoding lookup table from `fit_df`."""
        te_col = f"{col}_te"
        return (
            fit_df.group_by(col)
            .agg([
                pl.col("clk").sum().alias("_pos"),
                pl.col("clk").count().alias("_count"),
            ])
            .with_columns(
                ((pl.col("_pos") + self.smoothing * prior) / (pl.col("_count") + self.smoothing))
                .cast(pl.Float32)
                .alias(te_col)
            )
            .select([col, te_col])
        )

    def fit_target_encoding(self, train_df: pl.DataFrame) -> "CTRFeatureEngineer":
        """Fit smoothed target-encoding lookup tables on the full training partition."""
        logger.info(
            f"Fitting smoothed target encodings on train partition for: {self.target_encode_cols}"
        )
        self.global_ctr = float(train_df.select(pl.col("clk").mean()).item())

        self.target_encoding_maps = {
            col: self._fit_te_lookup(train_df, col, self.global_ctr)
            for col in self.target_encode_cols
        }
        return self

    def transform_target_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the train-fitted target-encoding maps; unseen categories fall back to global CTR."""
        if not self.target_encoding_maps:
            raise RuntimeError("fit_target_encoding() must be called before transform_target_encoding().")

        for col in self.target_encode_cols:
            te_col = f"{col}_te"
            df = df.join(self.target_encoding_maps[col], on=col, how="left").with_columns(
                pl.col(te_col).fill_null(self.global_ctr)
            )
        return df

    def add_target_encoding_oof(self, train_df: pl.DataFrame) -> pl.DataFrame:
        """Compute out-of-fold target encodings for train so a row's label never leaks into its own encoding."""
        logger.info(f"Computing {self.n_folds}-fold OOF target encodings on train partition...")
        n = train_df.height
        rng = np.random.RandomState(self.random_seed)
        fold_ids = rng.randint(0, self.n_folds, size=n)
        train_df = train_df.with_columns(pl.Series("_fold", fold_ids, dtype=pl.Int32))

        encoded_parts = []
        for fold in range(self.n_folds):
            fold_fit = train_df.filter(pl.col("_fold") != fold)
            fold_holdout = train_df.filter(pl.col("_fold") == fold)
            fold_prior = float(fold_fit.select(pl.col("clk").mean()).item())

            for col in self.target_encode_cols:
                te_col = f"{col}_te"
                stats = self._fit_te_lookup(fold_fit, col, fold_prior)
                fold_holdout = fold_holdout.join(stats, on=col, how="left").with_columns(
                    pl.col(te_col).fill_null(fold_prior)
                )
            encoded_parts.append(fold_holdout)

        result = pl.concat(encoded_parts).drop("_fold")
        return result

    # ------------------------------------------------------------------ #
    # Full Pipeline Orchestration
    # ------------------------------------------------------------------ #
    def fit_transform(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Run the full feature engineering pipeline across pre-split train/val/test partitions."""
        logger.info("=" * 60)
        logger.info("Starting CTR Feature Engineering Pipeline")
        logger.info("=" * 60)

        combined = pl.concat([
            train_df.with_columns(pl.lit("train").alias(_SPLIT_COL)),
            val_df.with_columns(pl.lit("val").alias(_SPLIT_COL)),
            test_df.with_columns(pl.lit("test").alias(_SPLIT_COL)),
        ])

        combined = self.add_exposure_sequence(combined)
        combined = self.add_cyclical_time_features(combined)
        combined = self.add_cross_features(combined)

        train_only = combined.filter(pl.col(_SPLIT_COL) == "train")
        self.fit_price_stats(train_only)
        combined = self.add_price_features(combined)

        train_fe = combined.filter(pl.col(_SPLIT_COL) == "train").drop(_SPLIT_COL)
        val_fe = combined.filter(pl.col(_SPLIT_COL) == "val").drop(_SPLIT_COL)
        test_fe = combined.filter(pl.col(_SPLIT_COL) == "test").drop(_SPLIT_COL)

        self.fit_target_encoding(train_fe)
        train_fe = self.add_target_encoding_oof(train_fe)
        val_fe = self.transform_target_encoding(val_fe)
        test_fe = self.transform_target_encoding(test_fe)

        logger.info("=" * 60)
        logger.info("CTR Feature Engineering Pipeline Completed Successfully!")
        logger.info("=" * 60)

        return train_fe, val_fe, test_fe
