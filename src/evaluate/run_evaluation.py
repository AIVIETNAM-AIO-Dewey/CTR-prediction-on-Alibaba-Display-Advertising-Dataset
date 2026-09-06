"""CLI for evaluating persisted CTR model artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.evaluate.evaluator import evaluate_suite, write_evaluation_outputs
from src.models.train import load_config

logger = logging.getLogger(__name__)

MODEL_KEYS = (
    "logistic_regression",
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fitted CTR artifacts on validation and test partitions."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--plots-dir", default="outputs/model_evaluation")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.30, 0.50],
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    model_specs: List[Dict[str, Any]] = []
    for model_key in args.models:
        config_path = Path(args.config_dir) / f"{model_key}.yaml"
        config = load_config(str(config_path))
        use_fe = config.get("data", {}).get("use_fe", True)
        basename = config.get("paths", {}).get("model_basename", model_key)
        artifact = Path(args.models_dir) / f"{basename}{'_fe' if use_fe else '_baseline'}.joblib"
        manifest = Path(args.experiments_dir) / f"{model_key}_run.json"
        model_specs.append(
            {
                "model_key": model_key,
                "config": config,
                "artifact_path": artifact,
                "manifest_path": manifest if manifest.exists() else None,
            }
        )

    results = evaluate_suite(
        model_specs=model_specs,
        processed_dir=args.processed_dir,
        thresholds=args.thresholds,
        sample_size=args.sample_size,
        sample_fraction=args.sample_fraction,
        random_seed=args.seed,
    )
    paths = write_evaluation_outputs(
        results,
        experiments_dir=args.experiments_dir,
        plots_dir=args.plots_dir,
        metadata={
            "processed_dir": str(args.processed_dir),
            "models_dir": str(args.models_dir),
            "sample_size": args.sample_size,
            "sample_fraction": args.sample_fraction,
            "random_seed": args.seed,
            "models": list(args.models),
        },
    )
    logger.info("Evaluation outputs: %s", json.dumps({key: str(value) for key, value in paths.items()}))


if __name__ == "__main__":
    main()
