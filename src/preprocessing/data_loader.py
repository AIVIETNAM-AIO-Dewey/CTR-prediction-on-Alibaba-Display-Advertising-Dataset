"""
Data Loader Module for CTR Prediction Dataset.
Provides high-performance, memory-efficient loading and lazy scanning of raw CSV files using Polars.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CTRDataLoader:
    """
    Handles reading and scanning raw dataset CSV files (user_profile, ad_feature, raw_sample)
    using Polars LazyFrame (scan_csv) and streaming execution for optimal memory efficiency.
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
    def _get_column_strip_map(columns: List[str]) -> Dict[str, str]:
        """Generate rename mapping to strip trailing/leading whitespace from headers."""
        return {col: col.strip() for col in columns if col != col.strip()}

    @classmethod
    def _scan_csv_stripped(cls, path: Path, **scan_csv_kwargs) -> pl.LazyFrame:
        """Lazily scan a CSV file and strip whitespace from its column headers."""
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")

        lf = pl.scan_csv(path, **scan_csv_kwargs)
        rename_map = cls._get_column_strip_map(lf.collect_schema().names())
        if rename_map:
            lf = lf.rename(rename_map)
        return lf

    def scan_user_profile(self) -> pl.LazyFrame:
        """
        Lazily scan user_profile.csv using Polars scan_csv.

        Returns:
            pl.LazyFrame: LazyFrame of user profile records.
        """
        return self._scan_csv_stripped(self.user_profile_path)

    def scan_ad_feature(self) -> pl.LazyFrame:
        """
        Lazily scan ad_feature.csv using Polars scan_csv.

        Returns:
            pl.LazyFrame: LazyFrame of ad metadata records.
        """
        return self._scan_csv_stripped(self.ad_feature_path)

    def scan_raw_sample(self) -> pl.LazyFrame:
        """
        Lazily scan raw_sample.csv with lazy sampling and streaming optimization.

        Returns:
            pl.LazyFrame: LazyFrame of interaction logs.
        """
        # If a fixed sample_size is requested, use n_rows in scan_csv to read only required rows
        if self.sample_size is not None and self.sample_size > 0:
            logger.info(f"Lazily scanning raw_sample with limit n_rows={self.sample_size:,}...")
            return self._scan_csv_stripped(self.raw_sample_path, n_rows=self.sample_size)
        return self._scan_csv_stripped(self.raw_sample_path)

    def load_user_profile(self) -> pl.DataFrame:
        """
        Load user_profile.csv into memory using lazy scan and collect.

        Returns:
            pl.DataFrame: Cleaned user profile records.
        """
        logger.info(f"Loading user profiles from: {self.user_profile_path}")
        df = self.scan_user_profile().collect()
        logger.info(f"Loaded user_profile: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_ad_feature(self) -> pl.DataFrame:
        """
        Load ad_feature.csv into memory using lazy scan and collect.

        Returns:
            pl.DataFrame: Cleaned ad feature records.
        """
        logger.info(f"Loading ad features from: {self.ad_feature_path}")
        df = self.scan_ad_feature().collect()
        logger.info(f"Loaded ad_feature: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_raw_sample(self) -> pl.DataFrame:
        """
        Load raw_sample.csv efficiently using scan_csv with lazy sampling / head limits.

        Returns:
            pl.DataFrame: Impression interaction log records.
        """
        logger.info(f"Loading interaction logs from: {self.raw_sample_path}")
        lf = self.scan_raw_sample()

        if self.sample_size is not None and self.sample_size > 0:
            logger.info(f"Collecting sample of {self.sample_size:,} rows...")
            df = lf.head(self.sample_size).collect()
        elif self.sample_fraction is not None and 0.0 < self.sample_fraction < 1.0:
            logger.info(f"Streaming and sampling {self.sample_fraction:.2%} of dataset (seed={self.random_seed})...")
            # Collect streaming and apply fraction sample
            df = lf.collect(streaming=True).sample(fraction=self.sample_fraction, seed=self.random_seed)
        else:
            logger.info("Collecting full raw_sample dataset using streaming engine...")
            df = lf.collect(streaming=True)

        logger.info(f"Loaded raw_sample: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_all(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Load all three datasets into memory.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
                (raw_sample_df, user_profile_df, ad_feature_df)
        """
        user_df = self.load_user_profile()
        ad_df = self.load_ad_feature()
        raw_df = self.load_raw_sample()
        return raw_df, user_df, ad_df
