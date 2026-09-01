"""
Kaggle Dataset Downloader using KaggleHub.
Downloads and transfers dataset files to the target data directory.
"""

from pathlib import Path
from typing import Optional, Union
import argparse
import logging
import os
import shutil
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = "pavansanagapati/ad-displayclick-data-on-taobaocom"
# Default destination directory: /home/hoangLD/Desktop/AIVIETNAM/Module-03/conquer/data
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data"


class KaggleHubDownloader:
    """
    Downloads datasets using kagglehub and syncs them to the target project directory.
    """

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    ):
        """
        Initialize the downloader.

        Args:
            dataset_name: Kaggle dataset slug (e.g. 'owner/dataset-name').
            output_dir: Directory where dataset files will be placed.
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir).resolve()

    def download_and_sync(self, force: bool = False, use_symlink: bool = False) -> Path:
        """
        Download dataset via kagglehub and copy/link files to output_dir.

        Args:
            force: Overwrite existing files in output directory.
            use_symlink: Create symlinks instead of copying (saves disk space).

        Returns:
            Path: Target output directory containing the dataset files.
        """
        try:
            import kagglehub
        except ImportError:
            logger.error("The 'kagglehub' package is not installed.")
            logger.error("Install it with: pip install kagglehub")
            raise

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Target dataset directory: {self.output_dir}")

        existing_csvs = list(self.output_dir.glob("*.csv"))
        if existing_csvs and not force:
            logger.info(f"Found {len(existing_csvs)} existing CSV file(s) in {self.output_dir}:")
            for f in existing_csvs:
                logger.info(f"  - {f.name} ({f.stat().st_size / (1024 * 1024):.2f} MB)")
            logger.info("Files already exist. Use --force to re-sync.")
            return self.output_dir

        logger.info(f"Downloading dataset '{self.dataset_name}' using kagglehub...")
        cached_path_str = kagglehub.dataset_download(self.dataset_name)
        cached_path = Path(cached_path_str).resolve()
        logger.info(f"KaggleHub cache directory: {cached_path}")

        # Sync files from cache to target output directory
        for item in cached_path.rglob("*"):
            if item.is_file():
                dest_file = self.output_dir / item.name
                if use_symlink:
                    if dest_file.exists() or dest_file.is_symlink():
                        dest_file.unlink()
                    dest_file.symlink_to(item)
                    logger.info(f"  -> Linked: {item.name}")
                else:
                    if dest_file.exists() and not force:
                        continue
                    shutil.copy2(item, dest_file)
                    size_mb = dest_file.stat().st_size / (1024 * 1024)
                    logger.info(f"  -> Copied: {item.name} ({size_mb:.2f} MB)")

        logger.info(f"✅ Successfully prepared dataset files in: {self.output_dir}")
        return self.output_dir


def download_dataset(
    dataset_name: str = DEFAULT_DATASET,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    use_symlink: bool = False,
) -> Path:
    """Convenience function to download and sync dataset."""
    downloader = KaggleHubDownloader(dataset_name=dataset_name, output_dir=output_dir)
    return downloader.download_and_sync(force=force, use_symlink=use_symlink)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Alibaba Display Advertising Dataset via KaggleHub."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset handle (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying files to save disk space.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-syncing files even if they already exist.",
    )

    args = parser.parse_args()

    try:
        download_dataset(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            force=args.force,
            use_symlink=args.symlink,
        )
    except Exception as exc:
        logger.error(f"Execution failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
