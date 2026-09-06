"""
CLI Entry Point: Fit the LightGBM CTR Model.

Trains ONLY LightGBM, from `configs/lightgbm.yaml`.
Stops at the artifact and computes no metrics: evaluation is Task 4 (src/evaluate/) and tuning
is Task 5 (experiments/tune_optuna.py).

Usage:
    python -m src.models.run_lightgbm                         # sample size from the config
    python -m src.models.run_lightgbm --sample-size 0         # full engineered dataset
    python -m src.models.run_lightgbm --sample-fraction 0.05  # 5% of every partition
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.train import run_cli


def main() -> None:
    run_cli(
        "lightgbm",
        "Fit the LightGBM CTR model (histogram binning and leaf-wise boosting).",
    )


if __name__ == "__main__":
    main()
