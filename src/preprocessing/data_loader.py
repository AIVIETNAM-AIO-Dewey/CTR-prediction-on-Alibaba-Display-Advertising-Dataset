"""
Data Loader Module for CTR Prediction Dataset.
Provides high-performance, memory-efficient loading of raw CSV files using Polars.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CTRDataLoader:
    """
    Handles reading raw dataset CSV files (user_profile, ad_feature, raw_sample)
    with memory optimization, whitespace stripping, and optional sampling.
    """

    def __init__(
        self,
        raw_user_profile_path: Union[str, Path] = "data/raw/user_profile.csv",
        raw_ad_feature_path: Union[str, Path] = "data/raw/ad_feature.csv",
        raw_sample_path: Union[str, Path] = "data/raw/raw_sample.csv",
        sample_fraction: Optional[float] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the CTRDataLoader.

        Args:
            raw_user_profile_path: Path to user_profile.csv.
            raw_ad_feature_path: Path to ad_feature.csv.
            raw_sample_path: Path to raw_sample.csv.
            sample_fraction: Optional fraction of raw_sample to sample (0.0 to 1.0).
            sample_size: Optional fixed number of rows to sample from raw_sample.
            random_seed: Random seed for reproducibility when sampling.
        """
        self.user_profile_path = Path(raw_user_profile_path)
        self.ad_feature_path = Path(raw_ad_feature_path)
        self.raw_sample_path = Path(raw_sample_path)
        self.sample_fraction = sample_fraction
        self.sample_size = sample_size
        self.random_seed = random_seed

    @staticmethod
    def _strip_column_names(df: pl.DataFrame) -> pl.DataFrame:
        """Strip trailing/leading whitespace from DataFrame column names."""
        return df.rename({col: col.strip() for col in df.columns})

    def load_user_profile(self) -> pl.DataFrame:
        """
        Load user_profile.csv with stripped column headers and optimized types.

        Returns:
            pl.DataFrame: Cleaned user profile records.
        """
        logger.info(f"Loading user profiles from: {self.user_profile_path}")
        if not self.user_profile_path.exists():
            raise FileNotFoundError(f"User profile file not found at: {self.user_profile_path}")

        df = pl.read_csv(self.user_profile_path)
        df = self._strip_column_names(df)
        logger.info(f"Loaded user_profile: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_ad_feature(self) -> pl.DataFrame:
        """
        Load ad_feature.csv with stripped column headers and optimized types.

        Returns:
            pl.DataFrame: Cleaned ad feature records.
        """
        logger.info(f"Loading ad features from: {self.ad_feature_path}")
        if not self.ad_feature_path.exists():
            raise FileNotFoundError(f"Ad feature file not found at: {self.ad_feature_path}")

        df = pl.read_csv(self.ad_feature_path)
        df = self._strip_column_names(df)
        logger.info(f"Loaded ad_feature: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_raw_sample(self) -> pl.DataFrame:
        """
        Load raw_sample.csv with stripped column headers and optional sampling.

        Returns:
            pl.DataFrame: Impression interaction log records.
        """
        logger.info(f"Loading interaction logs from: {self.raw_sample_path}")
        if not self.raw_sample_path.exists():
            raise FileNotFoundError(f"Raw sample file not found at: {self.raw_sample_path}")

        # If sample_size is requested, we can use Polars n_rows or sample after load
        df = pl.read_csv(self.raw_sample_path)
        df = self._strip_column_names(df)

        if self.sample_size is not None and self.sample_size < len(df):
            logger.info(f"Sampling {self.sample_size:,} rows (random_seed={self.random_seed})...")
            df = df.sample(n=self.sample_size, seed=self.random_seed)
        elif self.sample_fraction is not None and 0.0 < self.sample_fraction < 1.0:
            logger.info(f"Sampling {self.sample_fraction:.2%} of dataset (random_seed={self.random_seed})...")
            df = df.sample(fraction=self.sample_fraction, seed=self.random_seed)

        logger.info(f"Loaded raw_sample: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_all(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Load all three datasets.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
                (raw_sample_df, user_profile_df, ad_feature_df)
        """
        user_df = self.load_user_profile()
        ad_df = self.load_ad_feature()
        raw_df = self.load_raw_sample()
        return raw_df, user_df, ad_df
