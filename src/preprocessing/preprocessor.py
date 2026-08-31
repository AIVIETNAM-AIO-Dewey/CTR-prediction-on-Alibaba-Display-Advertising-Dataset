"""
Unified Preprocessor Pipeline for CTR Prediction.
Orchestrates data loading, cleaning, table joins, temporal extraction,
time-based dataset splitting, and saving to compressed Parquet format.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import logging
import polars as pl
import yaml

from src.preprocessing.cleaner import CTRDataCleaner
from src.preprocessing.data_loader import CTRDataLoader

logger = logging.getLogger(__name__)


class CTRPreprocessor:
    """
    End-to-End Preprocessing Pipeline for the Alibaba CTR Dataset.
    Pure preprocessing (loading, cleaning, merging, time parsing, and time-based splitting).
    """

    def __init__(
        self,
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        raw_user_profile_path: str = "data/raw/user_profile.csv",
        raw_ad_feature_path: str = "data/raw/ad_feature.csv",
        raw_sample_path: str = "data/raw/raw_sample.csv",
        processed_dir: str = "data/processed",
        train_end_date: str = "2017-05-11",
        val_date: str = "2017-05-12",
        test_date: str = "2017-05-13",
        sample_fraction: Optional[float] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the CTRPreprocessor pipeline.

        Args:
            config: Optional YAML config filepath or dictionary overriding defaults.
            raw_user_profile_path: Path to user_profile.csv.
            raw_ad_feature_path: Path to ad_feature.csv.
            raw_sample_path: Path to raw_sample.csv.
            processed_dir: Destination directory for processed files.
            train_end_date: Date dividing train set (inclusive).
            val_date: Date for validation set.
            test_date: Date for test set.
            sample_fraction: Fraction to sample for rapid testing.
            sample_size: Number of rows to sample.
            random_seed: Seed for sampling reproducibility.
        """
        self.config = self._load_config(config)

        # Extract parameters (prefer config if provided)
        paths_cfg = self.config.get("paths", {})
        split_cfg = self.config.get("split", {})
        impute_cfg = self.config.get("imputation", {})
        sample_cfg = self.config.get("sampling", {})

        self.user_profile_path = paths_cfg.get("raw_user_profile", raw_user_profile_path)
        self.ad_feature_path = paths_cfg.get("raw_ad_feature", raw_ad_feature_path)
        self.raw_sample_path = paths_cfg.get("raw_sample", raw_sample_path)
        self.processed_dir = Path(paths_cfg.get("processed_dir", processed_dir))

        self.train_end_date = split_cfg.get("train_end_date", train_end_date)
        self.val_date = split_cfg.get("val_date", val_date)
        self.test_date = split_cfg.get("test_date", test_date)

        # Sampling logic
        if sample_fraction is not None:
            self.sample_fraction = sample_fraction
        elif sample_cfg.get("enabled", False):
            self.sample_fraction = sample_cfg.get("sample_fraction")
        else:
            self.sample_fraction = None

        self.sample_size = sample_size or sample_cfg.get("sample_size")
        self.random_seed = sample_cfg.get("random_seed", random_seed)

        # Instantiate sub-components
        self.loader = CTRDataLoader(
            raw_user_profile_path=self.user_profile_path,
            raw_ad_feature_path=self.ad_feature_path,
            raw_sample_path=self.raw_sample_path,
            sample_fraction=self.sample_fraction,
            sample_size=self.sample_size,
            random_seed=self.random_seed,
        )

        self.cleaner = CTRDataCleaner(
            user_missing_val=impute_cfg.get("user_missing_val", -1),
            brand_unknown_val=impute_cfg.get("brand_unknown_val", -1),
            price_fallback=impute_cfg.get("price_fallback", "median"),
        )

        # Ensure processed output directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

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

    def extract_temporal_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse Unix timestamp and extract primary date and time components.

        Args:
            df: Polars DataFrame containing 'time_stamp' column.

        Returns:
            pl.DataFrame: DataFrame with datetime, date, hour, day_of_week, and is_weekend.
        """
        logger.info("Extracting temporal fields from Unix timestamps...")

        df = df.with_columns([
            pl.from_epoch(pl.col("time_stamp"), time_unit="s").alias("datetime")
        ]).with_columns([
            pl.col("datetime").dt.date().cast(pl.Utf8).alias("date"),
            pl.col("datetime").dt.hour().cast(pl.UInt8).alias("hour"),
            pl.col("datetime").dt.weekday().cast(pl.UInt8).alias("day_of_week"),
        ]).with_columns([
            pl.when(pl.col("day_of_week") >= 6).then(1).otherwise(0).cast(pl.UInt8).alias("is_weekend")
        ])

        return df

    def split_by_time(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split dataset into Train, Validation, and Test sets based on time_stamp.
        Prevents future data leakage.

        Args:
            df: Merged and temporal-extracted DataFrame.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: (train_df, val_df, test_df)
        """
        logger.info("Splitting dataset by time (Train / Val / Test)...")

        # Check unique dates in data
        unique_dates = df.select(pl.col("date").unique()).to_series().to_list()
        unique_dates = sorted(unique_dates)
        logger.info(f"Available dates in data: {unique_dates}")

        # If date strings match expected range
        if (
            self.test_date in unique_dates
            and self.val_date in unique_dates
        ):
            train_df = df.filter(pl.col("date") <= self.train_end_date)
            val_df = df.filter(pl.col("date") == self.val_date)
            test_df = df.filter(pl.col("date") == self.test_date)
        else:
            # Fallback for sub-sampled data without all date boundaries:
            # Chronological 70% Train / 15% Val / 15% Test based on time_stamp quantiles
            logger.info("Using quantile chronological split (70% Train, 15% Val, 15% Test)...")
            sorted_df = df.sort("time_stamp")
            n = len(sorted_df)
            train_idx = int(0.70 * n)
            val_idx = int(0.85 * n)

            train_df = sorted_df.slice(0, train_idx)
            val_df = sorted_df.slice(train_idx, val_idx - train_idx)
            test_df = sorted_df.slice(val_idx, n - val_idx)

        logger.info(
            f"Split sizes -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
        )
        return train_df, val_df, test_df

    def run_pipeline(
        self, save_output: bool = True
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Execute full end-to-end preprocessing pipeline.

        Args:
            save_output: Whether to persist train/val/test splits to Parquet files.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: (train_df, val_df, test_df)
        """
        logger.info("=" * 60)
        logger.info("Starting CTR Data Preprocessing Pipeline")
        logger.info("=" * 60)

        # 1. Load Raw Data
        raw_df, user_df, ad_df = self.loader.load_all()

        # 2. Clean Individual Tables
        user_clean = self.cleaner.clean_user_profile(user_df)
        ad_clean = self.cleaner.clean_ad_feature(ad_df)
        raw_clean = self.cleaner.clean_raw_sample(raw_df)

        # 3. Merge Tables
        merged_df = self.cleaner.merge_tables(raw_clean, user_clean, ad_clean)

        # 4. Extract Temporal Fields
        processed_df = self.extract_temporal_fields(merged_df)

        # 5. Split by Time
        train_df, val_df, test_df = self.split_by_time(processed_df)

        # 6. Save Processed Data
        if save_output:
            self.save_processed_data(train_df, val_df, test_df)

        logger.info("=" * 60)
        logger.info("CTR Data Preprocessing Pipeline Completed Successfully!")
        logger.info("=" * 60)

        return train_df, val_df, test_df

    def save_processed_data(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> None:
        """
        Persist train, validation, and test splits into compressed Parquet files.

        Args:
            train_df: Processed training DataFrame.
            val_df: Processed validation DataFrame.
            test_df: Processed test DataFrame.
        """
        train_path = self.processed_dir / "train.parquet"
        val_path = self.processed_dir / "val.parquet"
        test_path = self.processed_dir / "test.parquet"
        meta_path = self.processed_dir / "metadata.json"

        logger.info(f"Saving train split to: {train_path}")
        train_df.write_parquet(train_path, compression="snappy")

        logger.info(f"Saving val split to: {val_path}")
        val_df.write_parquet(val_path, compression="snappy")

        logger.info(f"Saving test split to: {test_path}")
        test_df.write_parquet(test_path, compression="snappy")

        # Save metadata summary
        metadata = {
            "num_train_rows": len(train_df),
            "num_val_rows": len(val_df),
            "num_test_rows": len(test_df),
            "columns": train_df.columns,
            "train_ctr": float(train_df.select(pl.col("clk").mean()).item() * 100),
            "val_ctr": float(val_df.select(pl.col("clk").mean()).item() * 100),
            "test_ctr": float(test_df.select(pl.col("clk").mean()).item() * 100),
            "median_imputed_price": self.cleaner.median_price,
            "user_missing_val": self.cleaner.user_missing_val,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved preprocessing metadata to: {meta_path}")
        logger.info(f"Train CTR: {metadata['train_ctr']:.3f}% | Val CTR: {metadata['val_ctr']:.3f}% | Test CTR: {metadata['test_ctr']:.3f}%")
