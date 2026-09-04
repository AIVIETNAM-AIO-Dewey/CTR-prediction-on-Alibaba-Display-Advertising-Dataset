"""
CLI Entry Point: Fit the XGBoost CTR Model.

Trains ONLY XGBoost, from `configs/xgboost.yaml`; CatBoost has its own entry point and config.
Stops at the artifact and computes no metrics: evaluation is Task 4 (src/evaluate/) and tuning
is Task 5 (experiments/tune_optuna.py). See CONTRIBUTING.md.

Usage:
    python -m src.models.run_xgboost                         # sample size from the config
    python -m src.models.run_xgboost --sample-size 0         # full engineered dataset
    python -m src.models.run_xgboost --sample-fraction 0.05  # 5% of every partition
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.train import run_cli


def main() -> None:
    run_cli(
        "xgboost",
        "Fit the XGBoost CTR model (histogram trees with native categorical splits).",
    )


if __name__ == "__main__":
    main()
