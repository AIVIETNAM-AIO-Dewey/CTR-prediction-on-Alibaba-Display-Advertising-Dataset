"""
Command-Line Interface for running CTR Data Preprocessing.

Usage Examples:
    # Run with default config
    python -m src.preprocessing.run_preprocessing

    # Run on a 5% sample for quick testing
    python -m src.preprocessing.run_preprocessing --sample-fraction 0.05

    # Run on 100,000 rows sample
    python -m src.preprocessing.run_preprocessing --sample-size 100000

    # Run on full dataset with custom config
    python -m src.preprocessing.run_preprocessing --config configs/preprocessing.yaml --full
"""

import argparse
import sys
import time
import logging
from pathlib import Path

# Ensure root workspace is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.preprocessor import CTRPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_preprocessing")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CTR Data Preprocessing Pipeline (Loading, Cleaning, Merging, Temporal Splitting)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Fraction of raw sample dataset to process (e.g., 0.05 for 5%%).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of rows to sample from raw_sample for quick testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where processed Parquet files will be stored.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process the complete full raw dataset (ignores sampling).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sample_fraction = args.sample_fraction
    sample_size = args.sample_size

    if args.full:
        logger.info("Executing on FULL dataset (sampling disabled).")
        sample_fraction = None
        sample_size = None

    preprocessor = CTRPreprocessor(
        config=args.config,
        processed_dir=args.output_dir if args.output_dir else "data/processed",
        sample_fraction=sample_fraction,
        sample_size=sample_size,
    )

    start_time = time.time()
    train_df, val_df, test_df = preprocessor.run_pipeline(save_output=True)
    elapsed_time = time.time() - start_time

    logger.info(f"Total Preprocessing Execution Time: {elapsed_time:.2f} seconds.")
    logger.info("Output Datasets Summary:")
    logger.info(f"  • Train: {len(train_df):,} rows, {train_df.shape[1]} columns")
    logger.info(f"  • Val:   {len(val_df):,} rows, {val_df.shape[1]} columns")
    logger.info(f"  • Test:  {len(test_df):,} rows, {test_df.shape[1]} columns")


if __name__ == "__main__":
    main()
