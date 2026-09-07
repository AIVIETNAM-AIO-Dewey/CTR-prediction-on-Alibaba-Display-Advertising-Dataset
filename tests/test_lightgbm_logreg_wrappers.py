"""Regression tests for the LightGBM and logistic-regression setup."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.models import (
    LightGBMCTRModel,
    LightGBMModel,
    LogisticRegressionCTRModel,
    LogisticRegressionModel,
)
from src.models.data_utils import (
    ctr_partition_rows,
    iter_ctr_partition_batches,
    load_ctr_dataset,
)
from src.models.train import MODEL_REGISTRY, fit_from_config


def pandas_data():
    X_train = pd.DataFrame(
        {
            "price": [1.0, 2.0, np.nan, 4.0, 3.5, 1.5, 6.0, 2.5, 5.0, 7.0, 2.2, 6.5],
            "score": [0.1, 0.8, 0.2, np.nan, 0.7, 0.4, 0.95, 0.3, 0.85, 0.6, 0.15, 0.9],
            "pid": ["a", "b", None, "a", "b", "c", "c", "a", "b", None, "c", "a"],
        }
    )
    y_train = pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1])
    X_test = pd.DataFrame(
        {
            "price": [np.nan, 8.0, 1.2],
            "score": [0.25, np.nan, 0.5],
            "pid": ["new", None, "a"],
        }
    )
    return X_train, y_train, X_test


class PublicApiTests(unittest.TestCase):
    def test_lazy_exports_and_registry(self):
        self.assertIs(LightGBMModel, LightGBMCTRModel)
        self.assertIs(LogisticRegressionModel, LogisticRegressionCTRModel)
        self.assertIn("lightgbm", MODEL_REGISTRY)
        self.assertIn("logistic_regression", MODEL_REGISTRY)


class WrapperContractMixin:
    model_class = None

    def build_model(self):
        raise NotImplementedError

    def test_pandas_polars_schema_and_round_trip(self):
        X_train, y_train, X_test = pandas_data()
        model = self.build_model()
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(pl.from_pandas(X_test))
        self.assertEqual(probabilities.shape, (len(X_test),))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((0 <= probabilities) & (probabilities <= 1)).all())
        self.assertTrue(np.array_equal(model.predict(X_test), (probabilities >= 0.5)))

        reordered = X_test[["pid", "score", "price"]]
        np.testing.assert_allclose(probabilities, model.predict_proba(reordered))
        with self.assertRaisesRegex(ValueError, "Missing required feature"):
            model.predict_proba(X_test.drop(columns=["price"]))
        with self.assertRaisesRegex(ValueError, "provided together"):
            self.build_model().fit(X_train, y_train, X_val=X_test)

        importance = model.get_feature_importance()
        self.assertGreater(len(importance), 0)
        numeric_columns = [
            column
            for column, dtype in importance.schema.items()
            if dtype.is_numeric()
        ]
        for column in numeric_columns:
            self.assertTrue(np.isfinite(importance.get_column(column).to_numpy()).all())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            restored = self.model_class.load(path)
            np.testing.assert_allclose(probabilities, restored.predict_proba(X_test))

    def test_save_before_fit_and_numeric_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "fitted before saving"):
                self.build_model().save(Path(directory) / "invalid.joblib")

        X = np.array(
            [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
        )
        y = np.array([0, 0, 0, 1, 1, 1])
        model = self.model_class(
            n_estimators=10, verbose=-1, verbose_eval=0
        ) if self.model_class is LightGBMModel else self.model_class(max_iter=100)
        model.fit(X, y)
        self.assertEqual(model.predict_proba(X).shape, (len(X),))


class LightGBMContractTests(WrapperContractMixin, unittest.TestCase):
    model_class = LightGBMModel

    def build_model(self):
        return LightGBMModel(
            categorical_features=["pid"],
            n_estimators=25,
            learning_rate=0.15,
            num_leaves=7,
            min_child_samples=1,
            n_jobs=1,
            verbose=-1,
            verbose_eval=0,
            early_stopping_rounds=5,
        )

    def test_validation_early_stopping_and_config(self):
        X_train, y_train, X_test = pandas_data()
        model = self.build_model()
        model.fit(X_train, y_train, X_val=X_test, y_val=pd.Series([0, 1, 0]))
        self.assertGreater(model.best_iteration_, 0)
        configured = LightGBMModel.from_config(
            {
                "params": {"n_estimators": 9, "early_stopping_rounds": 3, "verbose_eval": 0},
                "data": {"random_seed": 7},
                "features": {"categorical": ["pid"]},
            }
        )
        self.assertEqual(configured.early_stopping_rounds, 3)
        self.assertEqual(configured.random_state, 7)


class LogisticRegressionContractTests(WrapperContractMixin, unittest.TestCase):
    model_class = LogisticRegressionModel

    def build_model(self):
        return LogisticRegressionModel(
            categorical_features=["pid"],
            numeric_features=["price", "score"],
            max_iter=200,
            n_jobs=1,
        )

    def test_sgd_path(self):
        X_train, y_train, X_test = pandas_data()
        model = LogisticRegressionModel(
            categorical_features=["pid"],
            numeric_features=["price", "score"],
            use_sgd=True,
            max_iter=200,
        )
        model.fit(pl.from_pandas(X_train), pl.Series(y_train))
        self.assertTrue(np.isfinite(model.predict_proba(X_test)).all())


class DatasetAndRunnerTests(unittest.TestCase):
    def _write_partitions(self, directory: Path) -> None:
        train = pl.DataFrame(
            {
                "clk": [0, 1, 0, 1, 0, 1, 0, 1],
                "price": [1.0, 2.0, None, 4.0, 1.5, 5.0, 2.5, 6.0],
                "pid": ["a", "b", None, "a", "c", "b", "c", "a"],
                "nonclk": [1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        validation = train.head(4)
        test = train.tail(4)
        train.write_parquet(directory / "train_fe.parquet")
        validation.write_parquet(directory / "val_fe.parquet")
        test.write_parquet(directory / "test_fe.parquet")

    def test_loader_and_fit_only_shared_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_partitions(directory)
            dataset = load_ctr_dataset(
                processed_dir=str(directory),
                target_col="clk",
                exclude_cols=["nonclk"],
                categorical_cols=["pid"],
                numeric_cols=["price"],
                use_fe=True,
                sample_fraction=0.5,
            )
            self.assertEqual(len(dataset.X_train), 4)
            self.assertEqual(dataset.feature_names, ["price", "pid"])

            config = {
                "model": "logistic_regression",
                "paths": {
                    "processed_dir": str(directory),
                    "models_dir": str(directory / "models"),
                },
                "data": {
                    "use_fe": True,
                    "sample_size": None,
                    "sample_fraction": None,
                    "random_seed": 42,
                },
                "features": {
                    "target": "clk",
                    "exclude_cols": ["clk", "nonclk"],
                    "categorical": ["pid"],
                    "numeric": ["price"],
                    "drop_features": [],
                },
                "params": {"max_iter": 100, "n_jobs": 1},
            }
            result = fit_from_config(
                config=config,
                save_artifact=False,
                write_manifest=False,
            )
            self.assertTrue(result.model.is_fitted)
            self.assertFalse((directory / "models").exists())
            self.assertNotIn("metrics", result.manifest)

    def test_loader_rejects_conflicting_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_partitions(directory)
            with self.assertRaisesRegex(ValueError, "either sample_size or sample_fraction"):
                load_ctr_dataset(
                    processed_dir=str(directory),
                    sample_size=2,
                    sample_fraction=0.5,
                )

    def test_streaming_logistic_uses_all_rows_without_dataset_materialisation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_partitions(directory)
            batches = list(
                iter_ctr_partition_batches(
                    directory, "train", ["price", "pid"], target_col="clk", batch_size=2
                )
            )
            self.assertEqual(sum(len(labels) for _frame, labels in batches), 8)
            self.assertEqual(ctr_partition_rows(directory, "train"), 8)

            config = {
                "model": "logistic_regression",
                "paths": {"processed_dir": str(directory), "models_dir": str(directory / "models")},
                "data": {"use_fe": True, "sample_size": 0, "sample_fraction": None, "random_seed": 42},
                "features": {
                    "target": "clk", "exclude_cols": ["clk", "nonclk"],
                    "categorical": ["pid"], "numeric": ["price"], "drop_features": [],
                },
                "params": {
                    "use_sgd": True, "solver": "sgd", "streaming": True,
                    "stream_batch_size": 2, "stream_epochs": 1, "alpha": 1e-4,
                    "n_jobs": 1,
                },
            }
            result = fit_from_config(
                config=config, save_artifact=False, write_manifest=False, sample_size=0
            )
            self.assertIsNone(result.dataset)
            self.assertEqual(result.manifest["train_rows"], 8)
            self.assertEqual(result.manifest["rows_seen"], 8)
            self.assertEqual(result.manifest["fit_mode"], "streaming_sgd")
            self.assertTrue(np.isfinite(result.model.predict_proba(batches[0][0])).all())


if __name__ == "__main__":
    unittest.main()
