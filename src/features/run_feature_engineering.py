"""
Command-Line Interface for running CTR Feature Engineering.

Reads the preprocessed train/val/test Parquet partitions produced by
`src.preprocessing.run_preprocessing`, applies the CTRFeatureEngineer
pipeline, and writes engineered Parquet partitions plus updated metadata.

Usage Examples:
    # Run with default config
    python -m src.features.run_feature_engineering

    # Run with a custom config
    python -m src.features.run_feature_engineering --config configs/feature_engineering.yaml
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

# Ensure root workspace is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
import yaml

from src.features.feature_engineer import CTRFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_feature_engineering")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CTR Feature Engineering Pipeline (exposure sequence, price, "
        "cyclical time, cross features, out-of-fold target encoding)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature_engineering.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing train/val/test.parquet (default: from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where engineered Parquet files will be stored (default: from config).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config file not found at {path}, using defaults.")
    return {}


def main():
    args = parse_args()
    config = load_config(args.config)
    paths_cfg = config.get("paths", {})

    input_dir = Path(args.input_dir or paths_cfg.get("input_dir", "data/processed"))
    output_dir = Path(args.output_dir or paths_cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train.parquet"
    val_path = input_dir / "val.parquet"
    test_path = input_dir / "test.parquet"

    logger.info(f"Reading preprocessed partitions from: {input_dir}")
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    engineer = CTRFeatureEngineer(config=config)

    start_time = time.time()
    train_fe, val_fe, test_fe = engineer.fit_transform(train_df, val_df, test_df)
    elapsed_time = time.time() - start_time

    train_out = output_dir / "train_fe.parquet"
    val_out = output_dir / "val_fe.parquet"
    test_out = output_dir / "test_fe.parquet"

    logger.info(f"Saving engineered train split to: {train_out}")
    train_fe.write_parquet(train_out, compression="snappy")
    logger.info(f"Saving engineered val split to: {val_out}")
    val_fe.write_parquet(val_out, compression="snappy")
    logger.info(f"Saving engineered test split to: {test_out}")
    test_fe.write_parquet(test_out, compression="snappy")

    metadata = {
        "num_train_rows": train_fe.height,
        "num_val_rows": val_fe.height,
        "num_test_rows": test_fe.height,
        "columns": train_fe.columns,
        "target_encode_columns": engineer.target_encode_cols,
        "target_encoding_smoothing": engineer.smoothing,
        "target_encoding_n_folds": engineer.n_folds,
        "global_train_ctr": engineer.global_ctr,
        "global_median_price": engineer.global_median_price,
    }
    meta_path = output_dir / "feature_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature engineering metadata to: {meta_path}")

    logger.info(f"Total Feature Engineering Execution Time: {elapsed_time:.2f} seconds.")
    logger.info("Output Datasets Summary:")
    logger.info(f"  • Train: {train_fe.height:,} rows, {train_fe.shape[1]} columns")
    logger.info(f"  • Val:   {val_fe.height:,} rows, {val_fe.shape[1]} columns")
    logger.info(f"  • Test:  {test_fe.height:,} rows, {test_fe.shape[1]} columns")


if __name__ == "__main__":
    main()
