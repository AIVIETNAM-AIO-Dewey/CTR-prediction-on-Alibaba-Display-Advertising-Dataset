"""Tests for the shared evaluation suite and the Kaggle data handoff."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.evaluate.evaluator import (
    evaluate_artifact,
    evaluate_predictions,
    write_evaluation_outputs,
)
from src.evaluate.metrics import (
    compute_probability_metrics,
    compute_threshold_metrics,
    select_best_f1_threshold,
)
from src.features.feature_engineer import CTRFeatureEngineer
from src.models.catboost_model import CatBoostCTRModel
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
            self.assertTrue(payload["metadata"]["test"])


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

    def test_catboost_gpu_device_is_a_persisted_wrapper_option(self):
        model = CatBoostCTRModel(task_type="GPU", devices="0:1")
        self.assertEqual(model.devices, "0:1")


if __name__ == "__main__":
    unittest.main()
