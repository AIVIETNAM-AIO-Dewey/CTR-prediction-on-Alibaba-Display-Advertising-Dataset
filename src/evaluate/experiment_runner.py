"""Small, testable helpers shared by the Kaggle experiment notebook."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional

from src.evaluate.evaluator import ModelEvaluationResult
from src.evaluate.signatures import config_fingerprint, partition_fingerprint


def validate_engineered_cache(
    input_dir: str | Path,
    output_dir: str | Path,
    feature_config_path: str | Path,
    pipeline_revision: Optional[str] = None,
) -> tuple[bool, Optional[dict[str, Any]], dict[str, Any]]:
    """Validate output files, schemas, metadata and input/config identity."""
    output_root = Path(output_dir)
    metadata_path = output_root / "feature_metadata.json"
    output_files = [output_root / f"{split}_fe.parquet" for split in ("train", "val", "test")]
    input_fingerprint = partition_fingerprint(input_dir)
    if not all(path.exists() for path in output_files) or not metadata_path.exists():
        return False, None, input_fingerprint
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        output_fingerprint = partition_fingerprint(output_root, suffix="_fe")
        output_columns = {
            split: list(info["schema"]) for split, info in output_fingerprint.items()
        }
        columns = metadata.get("columns")
        rows_match = all(
            metadata.get(f"num_{split}_rows") == input_fingerprint[split]["rows"] == output_fingerprint[split]["rows"]
            for split in ("train", "val", "test")
        )
        schema_match = bool(columns) and all(
            output_columns[split] == columns for split in ("train", "val", "test")
        )
        dtype_match = bool(metadata.get("schema")) and all(
            info["schema"] == metadata["schema"] for info in output_fingerprint.values()
        )
        valid = (
            metadata.get("input_fingerprint") == input_fingerprint
            and metadata.get("config_fingerprint") == config_fingerprint(feature_config_path)
            and metadata.get("memory_bounded") is True
            and (pipeline_revision is None or metadata.get("pipeline_revision") == pipeline_revision)
            and rows_match
            and schema_match
            and dtype_match
        )
        return valid, metadata if valid else None, input_fingerprint
    except Exception:
        return False, None, input_fingerprint


def load_compatible_results(
    path: str | Path,
    run_signatures: Mapping[str, str],
) -> dict[str, ModelEvaluationResult]:
    """Read only model results whose per-model signature still matches."""
    result_path = Path(path)
    if not result_path.exists():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        compatible: dict[str, ModelEvaluationResult] = {}
        for item in payload.get("models", []):
            result = ModelEvaluationResult.from_dict(item)
            if run_signatures.get(result.model_key) == result.metadata.get("run_signature"):
                compatible[result.model_key] = result
        return compatible
    except Exception:
        return {}


def load_valid_checkpoint(
    model_key: str,
    artifact_path: str | Path,
    manifest_path: str | Path,
    model_loader: Callable[[str | Path], Any],
    expected_signature: str,
    expected_rows: Mapping[str, int],
) -> Any | None:
    """Load a checkpoint only when its manifest and model schema agree."""
    artifact = Path(artifact_path)
    manifest = Path(manifest_path)
    if not artifact.exists() or not manifest.exists():
        return None
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info.get("model") != model_key or info.get("run_signature") != expected_signature:
            return None
        if any(int(info.get(f"{split}_rows", -1)) != int(rows) for split, rows in expected_rows.items()):
            return None
        model = model_loader(artifact)
        if list(info.get("feature_names", [])) != list(model.feature_names):
            return None
        if int(info.get("n_features", -1)) != len(model.feature_names):
            return None
        return model
    except Exception:
        return None


def release_iteration_state(namespace: MutableMapping[str, Any]) -> None:
    """Drop large notebook locals before collecting cyclic references."""
    for name in (
        "run",
        "checkpoint",
        "model",
        "X_val",
        "X_test",
        "y_val",
        "y_test",
        "p_val",
        "p_test",
    ):
        if name in namespace:
            namespace[name] = None
    gc.collect()
