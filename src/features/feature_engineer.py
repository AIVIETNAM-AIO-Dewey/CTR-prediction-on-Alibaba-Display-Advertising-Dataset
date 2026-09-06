"""
Feature Engineering Module for CTR Prediction.

Implements exposure-sequence counters, price transformations, cyclical time
encodings, cross features, and out-of-fold smoothed Bayesian target encoding.
"""

from pathlib import Path
import gc
import shutil
import tempfile
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

    def _add_target_encoding_oof_column(
        self,
        result: pl.DataFrame,
        source: pl.DataFrame,
        col: str,
        fold_totals: pl.DataFrame,
        total_positive: int,
        total_rows: int,
    ) -> pl.DataFrame:
        """Add one OOF TE column using aggregate subtraction from ``source``."""
        te_col = f"{col}_te"
        global_stats = source.group_by(col).agg(
            [pl.col("clk").sum().alias("_global_pos"), pl.len().alias("_global_count")]
        )
        fold_stats = source.group_by([col, "_fold"]).agg(
            [pl.col("clk").sum().alias("_heldout_pos"), pl.len().alias("_heldout_count")]
        )
        denominator = pl.col("_global_count") - pl.col("_heldout_count")
        # The fold prior is fitted on every row outside the held-out fold, not
        # on the category rows outside that fold.
        prior_denominator = pl.lit(total_rows) - pl.col("_fold_count_total")
        fold_prior = pl.when(prior_denominator > 0).then(
            (pl.lit(total_positive) - pl.col("_fold_pos_total")).truediv(prior_denominator)
        ).otherwise(float(self.global_ctr or 0.0))
        return (
            result.join(global_stats, on=col, how="left")
            .join(fold_stats, on=[col, "_fold"], how="left")
            .join(fold_totals, on="_fold", how="left")
            .with_columns(
                pl.when(denominator > 0)
                .then(
                    (
                        (pl.col("_global_pos") - pl.col("_heldout_pos"))
                        + self.smoothing * fold_prior
                    ).truediv(denominator + self.smoothing)
                )
                .otherwise(fold_prior)
                .cast(pl.Float32)
                .alias(te_col)
            )
            .drop(
                [
                    "_global_pos",
                    "_global_count",
                    "_heldout_pos",
                    "_heldout_count",
                    "_fold_pos_total",
                    "_fold_count_total",
                ]
            )
        )

    def add_target_encoding_oof(self, train_df: pl.DataFrame) -> pl.DataFrame:
        """Compute out-of-fold target encodings for train so a row's label never leaks into its own encoding."""
        logger.info(f"Computing {self.n_folds}-fold OOF target encodings on train partition...")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2 for out-of-fold target encoding.")

        # Keep the established RandomState assignment for reproducibility, but
        # store only a compact fold column and never materialize n_folds filtered
        # copies of the training frame.
        rng = np.random.RandomState(self.random_seed)
        if self.n_folds <= np.iinfo(np.int8).max:
            fold_dtype, fold_polars_dtype = np.int8, pl.Int8
        elif self.n_folds <= np.iinfo(np.int16).max:
            fold_dtype, fold_polars_dtype = np.int16, pl.Int16
        else:
            fold_dtype, fold_polars_dtype = np.int32, pl.Int32
        fold_ids = rng.randint(0, self.n_folds, size=train_df.height).astype(fold_dtype)
        train_df = train_df.with_row_index("_te_row_id").with_columns(
            pl.Series("_fold", fold_ids, dtype=fold_polars_dtype)
        )
        fold_totals = train_df.group_by("_fold").agg(
            [pl.col("clk").sum().alias("_fold_pos_total"), pl.len().alias("_fold_count_total")]
        )
        total_positive = int(train_df.select(pl.col("clk").sum()).item() or 0)
        total_rows = int(train_df.height)
        result = train_df
        for col in self.target_encode_cols:
            result = self._add_target_encoding_oof_column(
                result, train_df, col, fold_totals, total_positive, total_rows
            )
        return result.sort("_te_row_id").drop(["_fold", "_te_row_id"])

    def fit_transform_partitioned(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Feature-engineer chronological partitions without concatenating all rows.

        This keeps only the current partition plus train-fitted lookup tables in
        memory.  The caller must validate that the partitions are chronological;
        exposure counters intentionally carry state from one partition to the next.
        """
        exposure_state: dict[str, pl.DataFrame] = {}

        def transform_exposure(frame: pl.DataFrame) -> pl.DataFrame:
            frame = frame.sort("time_stamp")
            for keys, name in (
                (["user", "adgroup_id"], "user_adgroup_exposure_seq"),
                (["user", "cate_id"], "user_cate_exposure_seq"),
            ):
                state = exposure_state.get(name)
                local = (pl.col("time_stamp").cum_count().over(keys) - 1).cast(pl.Int64)
                if state is None:
                    frame = frame.with_columns(local.cast(pl.Int32).alias(name))
                else:
                    state_col = f"_{name}_prior"
                    frame = (
                        frame.join(state.rename({"_count": state_col}), on=keys, how="left")
                        .with_columns(
                            (local + pl.col(state_col).fill_null(0))
                            .cast(pl.Int32)
                            .alias(name)
                        )
                        .drop(state_col)
                    )
                counts = frame.group_by(keys).agg(pl.len().alias("_count"))
                if state is None:
                    exposure_state[name] = counts
                else:
                    exposure_state[name] = (
                        pl.concat([state, counts], how="vertical_relaxed")
                        .group_by(keys)
                        .agg(pl.col("_count").sum())
                    )
            return frame

        train = transform_exposure(train_df)
        train = self.add_cross_features(self.add_cyclical_time_features(train))
        self.fit_price_stats(train)
        train = self.add_price_features(train)
        self.fit_target_encoding(train)
        train = self.add_target_encoding_oof(train)

        val = transform_exposure(val_df)
        val = self.add_price_features(
            self.add_cross_features(self.add_cyclical_time_features(val))
        )
        val = self.transform_target_encoding(val)

        test = transform_exposure(test_df)
        test = self.add_price_features(
            self.add_cross_features(self.add_cyclical_time_features(test))
        )
        test = self.transform_target_encoding(test)
        return train, val, test

    def fit_transform_partitioned_paths(
        self,
        train_path: str | Path,
        val_path: str | Path,
        test_path: str | Path,
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        """Run feature engineering one parquet partition at a time.

        The train frame is released before validation is loaded and validation is
        released before test is loaded.  This is the entry point used by Kaggle;
        the in-memory methods above remain available for small local workflows.
        """
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        for input_path in (train_path, val_path, test_path):
            if not Path(input_path).exists():
                raise FileNotFoundError(input_path)
        exposure_state: dict[str, pl.DataFrame] = {}
        previous_max_time: Any = None
        counts: Dict[str, int] = {}
        feature_columns: Optional[List[str]] = None

        def transform_exposure(frame: pl.DataFrame) -> pl.DataFrame:
            nonlocal previous_max_time
            if frame.is_empty():
                raise ValueError("Feature-engineering partitions must not be empty.")
            current_min, current_max = frame.select(
                [
                    pl.col("time_stamp").min().alias("_min_time"),
                    pl.col("time_stamp").max().alias("_max_time"),
                ]
            ).row(0)
            if previous_max_time is not None and current_min <= previous_max_time:
                raise ValueError(
                    "Expected chronological partitions train -> val -> test; "
                    f"found {current_min!r} after {previous_max_time!r}."
                )
            previous_max_time = current_max
            frame = frame.sort("time_stamp")
            for keys, name in (
                (["user", "adgroup_id"], "user_adgroup_exposure_seq"),
                (["user", "cate_id"], "user_cate_exposure_seq"),
            ):
                state = exposure_state.get(name)
                local = (pl.col("time_stamp").cum_count().over(keys) - 1).cast(pl.Int64)
                if state is None:
                    frame = frame.with_columns(local.cast(pl.Int32).alias(name))
                else:
                    state_col = f"_{name}_prior"
                    frame = (
                        frame.join(state.rename({"_count": state_col}), on=keys, how="left")
                        .with_columns(
                            (local + pl.col(state_col).fill_null(0))
                            .cast(pl.Int32)
                            .alias(name)
                        )
                        .drop(state_col)
                    )
                current_counts = frame.group_by(keys).agg(pl.len().alias("_count"))
                if state is None:
                    exposure_state[name] = current_counts
                else:
                    exposure_state[name] = (
                        pl.concat([state, current_counts], how="vertical_relaxed")
                        .group_by(keys)
                        .agg(pl.col("_count").sum())
                    )
            return frame

        def collect_partition(path: str | Path) -> pl.DataFrame:
            """Collect one lazy parquet scan using Polars' streaming engine."""
            scan = pl.scan_parquet(path)
            try:
                return scan.collect(engine="streaming")
            except TypeError:  # pragma: no cover - compatibility with old Polars
                return scan.collect(streaming=True)

        def write_partition(name: str, frame: pl.DataFrame, destination_root: Path) -> None:
            nonlocal feature_columns
            if feature_columns is None:
                feature_columns = list(frame.columns)
            elif list(frame.columns) != feature_columns:
                raise ValueError(f"Feature schema changed in {name} partition.")
            destination = destination_root / f"{name}_fe.parquet"
            frame.write_parquet(destination, compression="snappy")
            counts[name] = frame.height

        def add_oof_columns_staged(frame: pl.DataFrame, destination_root: Path) -> pl.DataFrame:
            """Compute OOF columns one at a time, releasing each intermediate."""
            if self.n_folds <= np.iinfo(np.int8).max:
                fold_dtype, fold_polars_dtype = np.int8, pl.Int8
            elif self.n_folds <= np.iinfo(np.int16).max:
                fold_dtype, fold_polars_dtype = np.int16, pl.Int16
            else:
                fold_dtype, fold_polars_dtype = np.int32, pl.Int32
            fold_ids = np.random.RandomState(self.random_seed).randint(
                0, self.n_folds, size=frame.height
            ).astype(fold_dtype)
            frame = frame.with_row_index("_te_row_id").with_columns(
                pl.Series(
                    "_fold",
                    fold_ids,
                    dtype=fold_polars_dtype,
                )
            )
            fold_totals = frame.group_by("_fold").agg(
                [
                    pl.col("clk").sum().alias("_fold_pos_total"),
                    pl.len().alias("_fold_count_total"),
                ]
            )
            total_positive = int(frame.select(pl.col("clk").sum()).item() or 0)
            total_rows = int(frame.height)
            for col in self.target_encode_cols:
                frame = self._add_target_encoding_oof_column(
                    frame, frame, col, fold_totals, total_positive, total_rows
                )
                staged = destination_root / f"train_oof_{col}.parquet"
                frame.write_parquet(staged, compression="snappy")
                del frame
                gc.collect()
                frame = pl.read_parquet(staged)
            return frame.sort("_te_row_id").drop(["_fold", "_te_row_id"])

        def transform_target_encoding_staged(
            frame: pl.DataFrame, name: str, destination_root: Path
        ) -> pl.DataFrame:
            """Apply one train-fitted lookup at a time through disk intermediates."""
            for col in self.target_encode_cols:
                te_col = f"{col}_te"
                frame = frame.join(self.target_encoding_maps[col], on=col, how="left").with_columns(
                    pl.col(te_col).fill_null(self.global_ctr)
                )
                staged = destination_root / f"{name}_te_{col}.parquet"
                frame.write_parquet(staged, compression="snappy")
                del frame
                gc.collect()
                frame = pl.read_parquet(staged)
            return frame

        # Keep all partially written files in an isolated directory.  A killed
        # or failed run therefore cannot leave a mixture of old and new
        # partitions in the cache; completed files are atomically promoted at
        # the very end.
        temporary_root = Path(tempfile.mkdtemp(prefix=".feature_engineering_", dir=output_root))
        try:
            train = transform_exposure(collect_partition(train_path))
            train = self.add_cross_features(self.add_cyclical_time_features(train))
            self.fit_price_stats(train)
            train = self.add_price_features(train)
            # Materialise the base stage to disk before the aggregate target
            # encoding stage so the largest intermediate can be released.
            train_base = temporary_root / "train_base.parquet"
            train.write_parquet(train_base, compression="snappy")
            del train
            gc.collect()
            train = pl.read_parquet(train_base)
            self.fit_target_encoding(train)
            train = add_oof_columns_staged(train, temporary_root)
            write_partition("train", train, temporary_root)
            del train
            gc.collect()

            val = transform_exposure(collect_partition(val_path))
            val = self.add_price_features(
                self.add_cross_features(self.add_cyclical_time_features(val))
            )
            val_base = temporary_root / "val_base.parquet"
            val.write_parquet(val_base, compression="snappy")
            del val
            gc.collect()
            val = transform_target_encoding_staged(pl.read_parquet(val_base), "val", temporary_root)
            write_partition("val", val, temporary_root)
            del val
            gc.collect()

            test = transform_exposure(collect_partition(test_path))
            test = self.add_price_features(
                self.add_cross_features(self.add_cyclical_time_features(test))
            )
            test_base = temporary_root / "test_base.parquet"
            test.write_parquet(test_base, compression="snappy")
            del test
            gc.collect()
            test = transform_target_encoding_staged(pl.read_parquet(test_base), "test", temporary_root)
            write_partition("test", test, temporary_root)
            del test
            gc.collect()

            for name in ("train", "val", "test"):
                source = temporary_root / f"{name}_fe.parquet"
                destination = output_root / f"{name}_fe.parquet"
                source.replace(destination)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

        return {
            "num_train_rows": counts["train"],
            "num_val_rows": counts["val"],
            "num_test_rows": counts["test"],
            "columns": feature_columns or [],
            "target_encode_columns": list(self.target_encode_cols),
            "target_encoding_smoothing": self.smoothing,
            "target_encoding_n_folds": self.n_folds,
            "target_encoding_fold_strategy": "random_state_aggregate",
            "global_train_ctr": self.global_ctr,
            "global_median_price": self.global_median_price,
        }

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
