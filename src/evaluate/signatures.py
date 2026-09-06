"""Deterministic signatures for Kaggle feature caches and model checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a source/config file in bounded chunks."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(paths: Sequence[str | Path]) -> dict[str, str]:
    """Return deterministic content hashes for code/config inputs."""
    return {
        str(Path(path).as_posix()): file_sha256(path)
        for path in sorted((Path(path) for path in paths), key=lambda item: item.as_posix())
    }


def parquet_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return a cheap, reproducible parquet identity without hashing all bytes."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    schema = pl.read_parquet_schema(parquet_path)
    rows = int(pl.scan_parquet(parquet_path).select(pl.len()).collect().item())
    return {
        "name": parquet_path.name,
        "size": parquet_path.stat().st_size,
        "rows": rows,
        "schema": {name: str(dtype) for name, dtype in schema.items()},
    }


def partition_fingerprint(directory: str | Path, suffix: str = "") -> dict[str, Any]:
    root = Path(directory)
    return {
        split: parquet_fingerprint(root / f"{split}{suffix}.parquet")
        for split in ("train", "val", "test")
    }


def config_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_versions(packages: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def build_run_signature(payload: Mapping[str, Any]) -> str:
    """Hash the complete run identity used by resume validation."""
    return sha256_text(canonical_json(payload))


def build_training_signature(payload: Mapping[str, Any]) -> str:
    """Hash only fit-time inputs; report/evaluation settings stay out of it."""
    return build_run_signature({"kind": "training", **dict(payload)})


def build_evaluation_signature(payload: Mapping[str, Any]) -> str:
    """Hash evaluation inputs, including the already validated training identity."""
    return build_run_signature({"kind": "evaluation", **dict(payload)})
