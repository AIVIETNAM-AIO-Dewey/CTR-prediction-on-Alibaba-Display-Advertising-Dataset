"""Contract smoke tests for the CatBoost, XGBoost, and Random Forest wrappers."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import polars as pl

from src.models import CatBoostCTRModel, RandomForestCTRModel, XGBoostCTRModel


class TreeModelWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X_train = pd.DataFrame(
            {
                "category": ["a", "b", None, "a", "b", "a", "b", "a", "b", "a", "b", "a"],
                "value": [0.1, 0.8, np.nan, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.2, 0.9, 0.1],
                "value_2": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        cls.y_train = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        cls.X_val = pd.DataFrame(
            {
                "category": ["a", "unseen", None, "b"],
                "value": [0.15, np.nan, 0.65, 0.85],
                "value_2": [0, 1, 0, 1],
            }
        )
        cls.y_val = np.array([0, 1, 1, 1], dtype=np.int8)

    def _new_models(self):
        return [
            CatBoostCTRModel(
                categorical_features=["category"],
                iterations=8,
                depth=3,
                learning_rate=0.1,
                eval_metric="Logloss",
                thread_count=1,
                verbose=0,
            ),
            XGBoostCTRModel(
                categorical_features=["category"],
                n_estimators=8,
                max_depth=3,
                learning_rate=0.1,
                min_child_weight=1,
                eval_metric=["logloss"],
                n_jobs=1,
                verbosity=0,
            ),
            RandomForestCTRModel(
                categorical_features=["category"],
                n_estimators=5,
                max_depth=4,
                min_samples_split=2,
                min_samples_leaf=1,
                max_samples=None,
                n_jobs=1,
            ),
        ]

    def test_public_exports(self):
        self.assertIsNotNone(CatBoostCTRModel)
        self.assertIsNotNone(XGBoostCTRModel)
        self.assertIsNotNone(RandomForestCTRModel)

    def test_fit_predict_importance_and_round_trip(self):
        for model in self._new_models():
            with self.subTest(model=model.model_name):
                fit_kwargs = {}
                if isinstance(model, XGBoostCTRModel):
                    fit_kwargs["verbose_eval"] = 0
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val, **fit_kwargs)

                probabilities = model.predict_proba(pl.from_pandas(self.X_val))
                self.assertEqual(probabilities.shape, (len(self.X_val),))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
                self.assertEqual(model.predict(self.X_val).shape, (len(self.X_val),))

                importance = model.get_feature_importance()
                self.assertEqual(set(importance["feature"].to_list()), set(model.feature_names))
                self.assertTrue(np.isfinite(importance["importance"].to_numpy()).all())

                with tempfile.TemporaryDirectory() as tmp:
                    artifact = Path(tmp) / "model.joblib"
                    model.save(artifact)
                    loaded = type(model).load(artifact)
                    loaded_probabilities = loaded.predict_proba(self.X_val)
                    np.testing.assert_allclose(probabilities, loaded_probabilities, rtol=1e-6, atol=1e-7)

    def test_numeric_numpy_input(self):
        X = self.X_train[["value", "value_2"]].fillna(0.0).to_numpy(dtype=np.float32)
        X_val = self.X_val[["value", "value_2"]].fillna(0.0).to_numpy(dtype=np.float32)
        for model in [
            CatBoostCTRModel(iterations=4, depth=2, thread_count=1, verbose=0),
            XGBoostCTRModel(n_estimators=4, max_depth=2, n_jobs=1, verbosity=0),
            RandomForestCTRModel(n_estimators=3, max_depth=2, n_jobs=1, max_samples=None),
        ]:
            with self.subTest(model=model.model_name):
                model.fit(X, self.y_train)
                probabilities = model.predict_proba(X_val)
                self.assertEqual(probabilities.shape, (len(X_val),))

    def test_invalid_state_and_schema(self):
        for model in self._new_models():
            with self.subTest(model=model.model_name):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(RuntimeError):
                        model.save(Path(tmp) / "unfitted.joblib")

                model.fit(self.X_train, self.y_train)
                missing = self.X_val.drop(columns=["value_2"])
                with self.assertRaises(ValueError):
                    model.predict_proba(missing)


if __name__ == "__main__":
    unittest.main()
