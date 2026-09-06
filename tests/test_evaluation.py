"""Tests for the shared evaluation suite and the Kaggle data handoff."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from src.evaluate.evaluator import (
    evaluate_artifact,
    evaluate_predictions,
    write_evaluation_outputs,
)
from src.evaluate.experiment_runner import (
    load_compatible_results,
    load_valid_checkpoint,
    release_iteration_state,
    validate_engineered_cache,
)
from src.evaluate.signatures import config_fingerprint, partition_fingerprint
from src.evaluate.metrics import (
    compute_probability_metrics,
    compute_threshold_metrics,
    select_best_f1_threshold,
)
from src.features.feature_engineer import CTRFeatureEngineer, _duckdb_table_columns
from src.models.catboost_model import CatBoostCTRModel
from src.models.data_utils import load_ctr_dataset
from src.models.logistic_regression_model import LogisticRegressionModel


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 1, 1])
        self.p = np.array([0.1, 0.2, 0.8, 0.9])

    def test_probability_metrics_are_finite(self):
        metrics = compute_probability_metrics(self.y, self.p)
        self.assertEqual(set(metrics), {"roc_auc", "log_loss", "pr_auc", "brier_score"})
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_threshold_metrics_and_validation_selection(self):
        rows = compute_threshold_metrics(self.y, self.p, [0.5, 0.8])
        self.assertEqual(rows[0]["true_positive"], 2)
        self.assertEqual(rows[0]["false_positive"], 0)
        threshold = select_best_f1_threshold(self.y, self.p)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_invalid_inputs_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "different lengths"):
            compute_probability_metrics(self.y, self.p[:2])
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_probability_metrics(self.y, [0.1, np.nan, 0.8, 0.9])
        with self.assertRaisesRegex(ValueError, "binary"):
            compute_probability_metrics([0, 2, 1, 0], self.p)
        with self.assertRaisesRegex(ValueError, "both classes"):
            compute_probability_metrics([0, 0], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "Threshold"):
            compute_threshold_metrics(self.y, self.p, [1.2])


class EvaluationOutputTests(unittest.TestCase):
    def test_validation_threshold_is_reused_on_test_and_outputs_round_trip(self):
        result = evaluate_predictions(
            "synthetic",
            self_y := np.array([0, 0, 1, 1]),
            np.array([0.1, 0.2, 0.8, 0.9]),
            np.array([0, 1, 0, 1]),
            np.array([0.2, 0.7, 0.3, 0.8]),
            [0.05, 0.5],
        )
        self.assertEqual(
            result.validation.selected_threshold,
            result.test.selected_threshold,
        )
        selected_test = [
            row for row in result.test.threshold_metrics if row["selected"]
        ]
        self.assertEqual(len(selected_test), 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_evaluation_outputs(
                [result],
                root / "experiments",
                root / "plots",
                metadata={"test": True},
            )
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["metrics_csv"].exists())
            self.assertTrue(paths["thresholds_csv"].exists())
            self.assertEqual(len(list((root / "plots").glob("*.png"))), 6)
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["models"][0]["model_key"], "synthetic")
            self.assertEqual(payload["models"][0]["rank"], 1)
            self.assertEqual(payload["best_model"], "synthetic")
            self.assertEqual(payload["ranking"][0]["model"], "synthetic")
            self.assertTrue(payload["metadata"]["test"])

            result.metadata["evaluation_signature"] = "sig"
            write_evaluation_outputs(
                [result], root / "experiments", root / "plots", metadata={"test": True}
            )
            compatible = load_compatible_results(
                root / "experiments" / "model_evaluation_results.json",
                {"synthetic": "sig"},
            )
            self.assertIn("synthetic", compatible)
            self.assertEqual(
                load_compatible_results(
                    root / "experiments" / "model_evaluation_results.json",
                    {"synthetic": "stale"},
                ),
                {},
            )


class FeatureEngineeringAndArtifactTests(unittest.TestCase):
    @staticmethod
    def _partition(rows: int, offset: int = 0) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "clk": [(index + offset) % 2 for index in range(rows)],
                "time_stamp": list(range(offset, offset + rows)),
                "user": [index % 3 for index in range(rows)],
                "adgroup_id": [index % 2 for index in range(rows)],
                "cate_id": [index % 4 for index in range(rows)],
                "price": [1.0 + index for index in range(rows)],
                "hour": [index % 24 for index in range(rows)],
                "day_of_week": [index % 7 for index in range(rows)],
                "final_gender_code": [index % 2 for index in range(rows)],
                "pid": [f"pid_{index % 3}" for index in range(rows)],
                "brand": [f"brand_{index % 2}" for index in range(rows)],
                "customer": [f"customer_{index % 3}" for index in range(rows)],
            }
        )

    def test_plain_partitions_build_engineered_features_without_leakage_columns(self):
        engineer = CTRFeatureEngineer(
            {"feature_engineering": {"target_encoding": {"n_folds": 2}}}
        )
        train, val, test = engineer.fit_transform(
            self._partition(8), self._partition(4, 8), self._partition(4, 12)
        )
        for frame in (train, val, test):
            self.assertIn("price_log", frame.columns)
            self.assertIn("cate_id_te", frame.columns)
            self.assertIn("gender_x_cate", frame.columns)
        self.assertNotIn("_fold", train.columns)

    def test_artifact_evaluation_loads_only_model_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = pd.DataFrame({"price": [1, 2, 3, 4, 5, 6], "clk": [0, 0, 0, 1, 1, 1]})
            val = pd.DataFrame({"price": [1.5, 2.5, 4.5, 5.5], "clk": [0, 1, 0, 1]})
            test = pd.DataFrame({"price": [1.2, 3.2, 4.2, 6.2], "clk": [0, 0, 1, 1]})
            pl.from_pandas(train).write_parquet(root / "train_fe.parquet")
            pl.from_pandas(val).write_parquet(root / "val_fe.parquet")
            pl.from_pandas(test).write_parquet(root / "test_fe.parquet")

            model = LogisticRegressionModel(
                numeric_features=["price"], max_iter=100, n_jobs=1
            )
            model.fit(train[["price"]], train["clk"])
            artifact = root / "logistic_regression_fe.joblib"
            model.save(artifact)
            config = {
                "model": "logistic_regression",
                "data": {"use_fe": True},
                "features": {"target": "clk"},
            }
            result = evaluate_artifact(
                "logistic_regression",
                config,
                artifact,
                root,
                thresholds=[0.1, 0.5],
            )
            self.assertEqual(result.validation.n_rows, 4)
            self.assertEqual(result.test.n_rows, 4)

    def test_dataset_loader_can_omit_validation_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, rows in (("train", 4), ("val", 2), ("test", 2)):
                frame = self._partition(rows)
                frame.write_parquet(root / f"{split}.parquet")
            dataset = load_ctr_dataset(
                processed_dir=root,
                use_fe=False,
                splits=("train",),
            )
            self.assertIsNone(dataset.X_val)
            self.assertIsNone(dataset.X_test)
            dataset = load_ctr_dataset(
                processed_dir=root,
                use_fe=False,
                splits=("train", "val"),
            )
            self.assertIsNotNone(dataset.X_val)
            self.assertIsNone(dataset.X_test)

    def test_dataset_loader_rejects_missing_requested_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._partition(4).write_parquet(root / "train.parquet")
            with self.assertRaisesRegex(FileNotFoundError, "val partition"):
                load_ctr_dataset(processed_dir=root, use_fe=False, splits=("train", "val"))

    def test_catboost_gpu_device_reaches_estimator_constructor(self):
        model = CatBoostCTRModel(task_type="GPU", devices="0:1", verbose=False)
        with patch("src.models.catboost_model.cb.CatBoostClassifier") as constructor:
            constructor.return_value.get_best_iteration.return_value = 0
            model.fit(pl.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}), [0, 0, 1, 1])
        self.assertEqual(constructor.call_args.kwargs["devices"], "0:1")

    def test_oof_target_encoding_matches_independent_reference(self):
        frame = self._partition(12)
        engineer = CTRFeatureEngineer(
            {
                "feature_engineering": {
                    "target_encoding": {
                        "columns": ["cate_id"],
                        "n_folds": 3,
                        "smoothing": 2.0,
                        "random_seed": 42,
                    }
                }
            }
        )
        actual = engineer.add_target_encoding_oof(frame)
        fold_ids = np.random.RandomState(42).randint(0, 3, size=len(frame))
        labels = np.asarray(frame["clk"].to_list(), dtype=float)
        categories = frame["cate_id"].to_list()
        total_positive = labels.sum()
        expected = []
        for index, (category, fold) in enumerate(zip(categories, fold_ids)):
            outside = fold_ids != fold
            category_mask = np.asarray([value == category for value in categories])
            heldout_category = category_mask & ~outside
            fit_category = category_mask & outside
            fit_count = int(fit_category.sum())
            fold_count = int(outside.sum())
            fold_prior = (total_positive - labels[~outside].sum()) / fold_count
            if fit_count:
                value = (
                    labels[fit_category].sum() + 2.0 * fold_prior
                ) / (fit_count + 2.0)
            else:
                value = fold_prior
            expected.append(value)
        np.testing.assert_allclose(actual["cate_id_te"].to_numpy(), expected, rtol=1e-6)

    def test_memory_bounded_feature_engineering_matches_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partitions = [self._partition(8), self._partition(4, 8), self._partition(4, 12)]
            paths = []
            for name, frame in zip(("train", "val", "test"), partitions):
                path = root / f"{name}.parquet"
                frame.write_parquet(path)
                paths.append(path)

            config = {"feature_engineering": {"target_encoding": {"n_folds": 2}}}
            expected = CTRFeatureEngineer(config).fit_transform(*partitions)
            output = root / "engineered"
            with patch("src.features.feature_engineer.pl.read_parquet", side_effect=AssertionError("partition must stay disk-backed")):
                CTRFeatureEngineer(config).fit_transform_partitioned_paths(*paths, output)
            actual = [pl.read_parquet(output / f"{name}_fe.parquet") for name in ("train", "val", "test")]
            self.assertTrue(all(expected_frame.equals(actual_frame) for expected_frame, actual_frame in zip(expected, actual)))
            self.assertEqual(list(output.glob(".feature_engineering_*")), [])

    def test_duckdb_table_columns_returns_describe_column_names(self):
        connection = duckdb.connect()
        try:
            connection.execute('CREATE TABLE "feature ""current" (row_id BIGINT, label VARCHAR, score DOUBLE)')
            self.assertEqual(
                _duckdb_table_columns(connection, 'feature "current'),
                ["row_id", "label", "score"],
            )
        finally:
            connection.close()

    def test_memory_bounded_feature_engineering_rejects_non_chronological_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._partition(4, 10)
            val = self._partition(4, 0)
            test = self._partition(4, 20)
            paths = []
            for name, frame in zip(("train", "val", "test"), (train, val, test)):
                path = root / f"{name}.parquet"
                frame.write_parquet(path)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "chronological"):
                CTRFeatureEngineer().fit_transform_partitioned_paths(*paths, root / "engineered")
            self.assertEqual(list((root / "engineered").glob(".feature_engineering_*")), [])

    def test_engineered_cache_checks_actual_files_and_config_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "feature.yaml"
            config_path.write_text("feature_engineering:\n  target_encoding:\n    n_folds: 2\n", encoding="utf-8")
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for split, rows in (("train", 4), ("val", 2), ("test", 2)):
                frame = self._partition(rows)
                frame.write_parquet(input_dir / f"{split}.parquet")
                frame.write_parquet(output_dir / f"{split}_fe.parquet")
            output_fingerprint = partition_fingerprint(output_dir, suffix="_fe")
            metadata = {
                "num_train_rows": 4,
                "num_val_rows": 2,
                "num_test_rows": 2,
                "columns": list(output_fingerprint["train"]["schema"]),
                "schema": output_fingerprint["train"]["schema"],
                "input_fingerprint": partition_fingerprint(input_dir),
                "config_fingerprint": config_fingerprint(config_path),
                "memory_bounded": True,
                "pipeline_revision": "rev",
            }
            (output_dir / "feature_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            valid, _, _ = validate_engineered_cache(input_dir, output_dir, config_path, "rev")
            self.assertTrue(valid)
            self._partition(1).write_parquet(output_dir / "val_fe.parquet")
            valid, _, _ = validate_engineered_cache(input_dir, output_dir, config_path, "rev")
            self.assertFalse(valid)

    def test_checkpoint_requires_signature_and_exact_feature_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = pd.DataFrame({"price": [1, 2, 3, 4], "clk": [0, 0, 1, 1]})
            model = LogisticRegressionModel(numeric_features=["price"], max_iter=100)
            model.fit(train[["price"]], train["clk"])
            artifact = root / "model.joblib"
            model.save(artifact)
            manifest = root / "manifest.json"
            base = {
                "model": "logistic_regression",
                "training_signature": "sig",
                "train_rows": 4,
                "val_rows": 2,
                "test_rows": 2,
                "n_features": 1,
                "feature_names": ["price"],
            }
            manifest.write_text(json.dumps(base), encoding="utf-8")
            loaded = load_valid_checkpoint(
                "logistic_regression", artifact, manifest, LogisticRegressionModel.load,
                expected_training_signature="sig",
                expected_rows={"train": 4, "val": 2, "test": 2},
            )
            self.assertIsNotNone(loaded)
            base["training_signature"] = "stale"
            manifest.write_text(json.dumps(base), encoding="utf-8")
            self.assertIsNone(load_valid_checkpoint(
                "logistic_regression", artifact, manifest, LogisticRegressionModel.load,
                "sig", {"train": 4, "val": 2, "test": 2},
            ))

    def test_checkpoint_rejects_manifest_before_loading_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.joblib"
            artifact.write_bytes(b"not a model")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "model": "logistic_regression",
                "training_signature": "sig",
                "train_rows": 4,
                "val_rows": 0,
                "test_rows": 0,
                "n_features": 1,
                "feature_names": ["price"],
                "params": {"max_iter": 10},
            }), encoding="utf-8")
            loader = Mock(side_effect=AssertionError("artifact must not be loaded"))
            self.assertIsNone(load_valid_checkpoint(
                "logistic_regression", artifact, manifest, loader,
                expected_training_signature="sig",
                expected_rows={"train": 4, "val": 0, "test": 0},
                expected_params={"max_iter": 11},
            ))
            loader.assert_not_called()

    def test_release_iteration_state_clears_large_locals(self):
        namespace = {name: object() for name in ("run", "model", "X_val", "X_test", "y_val", "y_test", "p_val", "p_test")}
        release_iteration_state(namespace)
        self.assertTrue(all(namespace[name] is None for name in namespace))

    def test_notebook_does_not_embed_token_in_clone_url(self):
        notebook = Path(__file__).parents[1] / "notebook" / "model_evaluation_experiments.ipynb"
        source = notebook.read_text(encoding="utf-8")
        self.assertNotIn('REPO_URL.replace("https://", f"https://{token}@")', source)
        self.assertNotIn("https://{token}@", source)


if __name__ == "__main__":
    unittest.main()
