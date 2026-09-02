"""
Data Cleaner Module for CTR Prediction Dataset.
Handles missing value imputation, schema consistency, data type casting, and table joins.
"""

from typing import Dict, Optional, Union
import logging
import polars as pl

logger = logging.getLogger(__name__)


class CTRDataCleaner:
    """
    Cleans raw demographic, ad feature, and interaction tables,
    handles missing values, and performs relational joins.
    """

    # Shared by clean_user_profile and merge_tables to avoid duplicated column lists.
    USER_DEMOGRAPHIC_DTYPES: Dict[str, pl.DataType] = {
        "cms_segid": pl.Int32,
        "cms_group_id": pl.Int32,
        "final_gender_code": pl.Int8,
        "age_level": pl.Int8,
        "pvalue_level": pl.Int8,
        "shopping_level": pl.Int8,
        "occupation": pl.Int8,
        "new_user_class_level": pl.Int8,
    }

    def __init__(
        self,
        user_missing_val: int = -1,
        brand_unknown_val: int = -1,
        price_fallback: str = "median",
    ):
        """
        Initialize CTRDataCleaner.

        Args:
            user_missing_val: Integer sentinel value used to impute missing demographic features.
            brand_unknown_val: Value representing unknown brand.
            price_fallback: Strategy for missing/zero prices ('median' or 'zero').
        """
        self.user_missing_val = user_missing_val
        self.brand_unknown_val = brand_unknown_val
        self.price_fallback = price_fallback
        self.median_price: Optional[float] = None

    def clean_user_profile(self, user_df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean user demographics and impute missing values.

        - pvalue_level (54.24% missing) -> imputed with user_missing_val
        - new_user_class_level (32.49% missing) -> imputed with user_missing_val
        - Cast demographic features to integer types safely.

        Args:
            user_df: Raw user profile Polars DataFrame.

        Returns:
            pl.DataFrame: Cleaned user profile DataFrame.
        """
        logger.info("Cleaning user_profile table...")

        demographic_exprs = [
            pl.col(col).cast(dtype, strict=False).fill_null(self.user_missing_val)
            for col, dtype in self.USER_DEMOGRAPHIC_DTYPES.items()
        ]

        user_clean = user_df.with_columns([
            pl.col("userid").cast(pl.UInt32, strict=False),
            *demographic_exprs,
        ])

        return user_clean

    def clean_ad_feature(self, ad_df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean ad metadata table and handle missing brand/price attributes.

        - Safely cast price to Float32 and compute valid median.
        - Impute missing / non-positive prices.
        - Safely cast brand to Int32 and handle brand == 0 / null as unknown brand (-1).

        Args:
            ad_df: Raw ad feature Polars DataFrame.

        Returns:
            pl.DataFrame: Cleaned ad feature DataFrame.
        """
        logger.info("Cleaning ad_feature table...")

        # Pre-cast price to Float32 safely
        price_col = pl.col("price").cast(pl.Float32, strict=False)

        # Calculate median positive price
        valid_prices = (
            ad_df.select(price_col.alias("price"))
            .filter(pl.col("price").is_not_null() & (pl.col("price") > 0))["price"]
        )
        if len(valid_prices) > 0:
            self.median_price = float(valid_prices.median())
        else:
            self.median_price = 0.0

        fallback_price_val = self.median_price if self.price_fallback == "median" else 0.0

        # Safe brand column expression
        brand_col = pl.col("brand").cast(pl.Int32, strict=False)

        ad_clean = ad_df.with_columns([
            pl.col("adgroup_id").cast(pl.UInt32, strict=False),
            pl.col("cate_id").cast(pl.UInt32, strict=False),
            pl.col("campaign_id").cast(pl.UInt32, strict=False),
            pl.col("customer").cast(pl.UInt32, strict=False),
            # Brand 0 or null is unknown in dataset
            pl.when(brand_col.is_null() | (brand_col == 0))
            .then(self.brand_unknown_val)
            .otherwise(brand_col)
            .cast(pl.Int32)
            .alias("brand"),
            # Clean and impute price
            pl.when(price_col.is_null() | (price_col <= 0))
            .then(fallback_price_val)
            .otherwise(price_col)
            .cast(pl.Float32)
            .alias("price"),
        ])

        return ad_clean

    def clean_raw_sample(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean raw interaction sample table.

        Args:
            raw_df: Raw interaction log Polars DataFrame.

        Returns:
            pl.DataFrame: Cleaned interaction DataFrame.
        """
        logger.info("Cleaning raw_sample table...")

        raw_clean = raw_df.with_columns([
            pl.col("user").cast(pl.UInt32, strict=False),
            pl.col("time_stamp").cast(pl.Int64, strict=False),
            pl.col("adgroup_id").cast(pl.UInt32, strict=False),
            pl.col("pid").cast(pl.Utf8),
            pl.col("clk").cast(pl.UInt8, strict=False),
            pl.col("nonclk").cast(pl.UInt8, strict=False),
        ])

        return raw_clean

    def merge_tables(
        self,
        raw_df: pl.DataFrame,
        user_df: pl.DataFrame,
        ad_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Merge raw interactions with user profiles and ad features via Left Joins.
        Imputes missing values for unmatched user IDs with user_missing_val.

        Args:
            raw_df: Cleaned interaction DataFrame.
            user_df: Cleaned user profile DataFrame.
            ad_df: Cleaned ad feature DataFrame.

        Returns:
            pl.DataFrame: Fully merged, unified dataset.
        """
        logger.info("Merging raw interactions with user profiles and ad features...")

        # 1. Left join with ad_feature on adgroup_id
        merged = raw_df.join(ad_df, on="adgroup_id", how="left")

        # 2. Left join with user_profile on user == userid
        merged = merged.join(user_df, left_on="user", right_on="userid", how="left")

        # 3. Impute demographics for unmatched users (approx 7% of raw users)
        impute_expressions = [
            pl.col(col).fill_null(self.user_missing_val).cast(dtype).alias(col)
            for col, dtype in self.USER_DEMOGRAPHIC_DTYPES.items()
            if col in merged.columns
        ]

        merged = merged.with_columns(impute_expressions)

        logger.info(f"Merged dataset shape: {merged.shape[0]:,} rows, {merged.shape[1]} columns")
        return merged
