"""One-model subprocess worker used by the Kaggle benchmark notebook."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Set these before importing sklearn/native model libraries.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from src.evaluate.evaluator import evaluate_artifact
from src.evaluate.experiment_runner import write_json_atomic
from src.models.train import fit_from_config
from src.models.data_utils import ctr_partition_rows
from src.models.train import get_model_class

logger = logging.getLogger(__name__)


def _thread_limited_environment() -> None:
    """Prevent native BLAS/OpenMP pools from multiplying a model's peak memory."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    _thread_limited_environment()
    model_key = str(job["model_key"])
    config = dict(job["config"])
    processed_dir = str(job["processed_dir"])
    models_dir = str(job["models_dir"])
    experiments_dir = Path(job["experiments_dir"])
    fit_sample_size = int(job.get("fit_sample_size", 0) or 0)
    if fit_sample_size != 0:
        raise ValueError("The full-data worker requires fit_sample_size=0; sampling is disabled.")
    started = time.time()
    print(
        f"[{model_key}] worker start mode={'streaming_sgd' if model_key == 'logistic_regression' else 'in_memory'} "
        f"fit_sample_size={fit_sample_size}", flush=True
    )
    basename = config.get("paths", {}).get("model_basename", model_key)
    artifact_path = Path(models_dir) / f"{basename}_fe.joblib"
    manifest_path = Path(config.get("paths", {}).get(
        "manifest_output", Path(job["experiments_dir"]) / f"{model_key}_run.json"
    ))
    model = None
    manifest: dict[str, Any] = {}
    if artifact_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_rows = ctr_partition_rows(processed_dir, "train", use_fe=True)
            if (
                manifest.get("training_signature") == job.get("training_signature")
                and int(manifest.get("train_rows", -1)) == expected_rows
                and manifest.get("sampling", {}).get("sample_size") is None
                and manifest.get("sampling", {}).get("sample_fraction") is None
            ):
                model = get_model_class(model_key).load(artifact_path)
                print(f"[{model_key}] valid checkpoint reused inside worker", flush=True)
        except Exception as exc:
            print(f"[{model_key}] checkpoint ignored: {exc}", flush=True)
            model = None
    if model is None:
        run = fit_from_config(
            config=config,
            config_path=config.get("source_path"),
            model_key=model_key,
            processed_dir=processed_dir,
            models_dir=models_dir,
            use_fe=True,
            sample_size=0,
            sample_fraction=None,
            random_seed=int(job.get("random_seed", 42)),
            training_signature=job.get("training_signature"),
            splits=job.get("train_splits"),
            save_artifact=True,
            write_manifest=True,
        )
        model = run.model
        artifact_path = run.artifact_path
        manifest_path = run.manifest_path or manifest_path
        manifest = run.manifest
    result = evaluate_artifact(
        model_key=model_key,
        config=config,
        artifact_path=artifact_path,
        processed_dir=processed_dir,
        thresholds=job.get("thresholds", [0.05, 0.10, 0.20, 0.30, 0.50]),
        sample_size=0,
        sample_fraction=None,
        random_seed=int(job.get("random_seed", 42)),
        manifest_path=manifest_path,
        batch_size=int(job.get("eval_batch_size", 65_536)),
    )
    result.metadata.update(
        {
            "training_signature": job.get("training_signature"),
            "evaluation_signature": job.get("evaluation_signature"),
            "elapsed_seconds": round(time.time() - started, 2),
        }
    )
    payload = {
        "status": "complete",
        "model_key": model_key,
        "result": result.to_dict(),
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
    }
    result_path = Path(job["result_path"])
    write_json_atomic(result_path, payload)
    print(
        f"[{model_key}] complete; rows={manifest.get('train_rows')} "
        f"val ROC-AUC={result.validation.probability_metrics['roc_auc']:.5f}", flush=True
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one full-data model job in isolation.")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        job = json.loads(args.job.read_text(encoding="utf-8"))
        run_job(job)
        return 0
    except Exception as exc:  # worker failures are persisted for notebook resume/reporting
        logger.exception("Model worker failed: %s", exc)
        try:
            job = json.loads(args.job.read_text(encoding="utf-8"))
            write_json_atomic(
                Path(job["result_path"]),
                {"status": "failed", "model_key": job.get("model_key"), "error": repr(exc)},
            )
        except Exception:
            logger.exception("Could not persist worker failure state")
        return 1


if __name__ == "__main__":
    sys.exit(main())
