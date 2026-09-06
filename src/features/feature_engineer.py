"""
Feature Engineering Module for CTR Prediction.

Implements exposure-sequence counters, price transformations, cyclical time
encodings, cross features, and out-of-fold smoothed Bayesian target encoding.
"""

from pathlib import Path
import gc
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union
import logging
import math

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

_SPLIT_COL = "_split"


class CTRFeatureEngineer:
    """Generates model-ready features for the Alibaba CTR dataset."""

    def __init__(
        self,
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        target_encode_cols: Optional[List[str]] = None,
        smoothing: float = 20.0,
        n_folds: int = 5,
        random_seed: int = 42,
    ):
        cfg = self._load_config(config)
        te_cfg = cfg.get("feature_engineering", {}).get("target_encoding", {})

        self.target_encode_cols = (
            target_encode_cols
            or te_cfg.get("columns")
            or ["cate_id", "brand", "customer", "pid"]
        )
        self.smoothing = te_cfg.get("smoothing", smoothing)
        self.n_folds = te_cfg.get("n_folds", n_folds)
        self.random_seed = te_cfg.get("random_seed", random_seed)

        self.cate_median_price: Optional[pl.DataFrame] = None
        self.global_median_price: Optional[float] = None
        self.global_ctr: Optional[float] = None
        self.target_encoding_maps: Dict[str, pl.DataFrame] = {}

    @staticmethod
    def _load_config(config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        """Load YAML config or return dictionary."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        config_path = Path(config)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"Config file not found at {config_path}, using defaults.")
        return {}

    # ------------------------------------------------------------------ #
    # Exposure Sequence (Ad Fatigue)
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_exposure_sequence(df: pl.DataFrame) -> pl.DataFrame:
        """Add prior-exposure counts per (user, adgroup_id) and (user, cate_id)."""
        logger.info("Computing exposure sequence counters (ad fatigue)...")
        df = df.sort("time_stamp")
        df = df.with_columns([
            (pl.col("time_stamp").cum_count().over(["user", "adgroup_id"]) - 1)
            .cast(pl.Int32)
            .alias("user_adgroup_exposure_seq"),
            (pl.col("time_stamp").cum_count().over(["user", "cate_id"]) - 1)
            .cast(pl.Int32)
            .alias("user_cate_exposure_seq"),
        ])
        return df

    # ------------------------------------------------------------------ #
    # Price Transformations
    # ------------------------------------------------------------------ #
    def fit_price_stats(self, train_df: pl.DataFrame) -> "CTRFeatureEngineer":
        """Fit per-category median price on the training partition."""
        logger.info("Fitting per-category median price statistics on train partition...")
        self.cate_median_price = (
            train_df.group_by("cate_id")
            .agg(pl.col("price").median().alias("cate_median_price"))
        )
        self.global_median_price = float(train_df.select(pl.col("price").median()).item() or 0.0)
        return self

    def add_price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add `price_log` (log1p) and `price_ratio_cate` (price / train-fitted category median)."""
        if self.cate_median_price is None:
            raise RuntimeError("fit_price_stats() must be called before add_price_features().")

        logger.info("Adding price_log and price_ratio_cate features...")
        fallback = self.global_median_price or 1.0

        df = df.join(self.cate_median_price, on="cate_id", how="left")
        df = df.with_columns([
            pl.col("price").log1p().cast(pl.Float32).alias("price_log"),
            pl.col("cate_median_price").fill_null(fallback).alias("cate_median_price"),
        ])
        df = df.with_columns([
            (
                pl.col("price")
                / pl.when(pl.col("cate_median_price") > 0)
                .then(pl.col("cate_median_price"))
                .otherwise(fallback)
            )
            .cast(pl.Float32)
            .alias("price_ratio_cate")
        ]).drop("cate_median_price")
        return df

    # ------------------------------------------------------------------ #
    # Cyclical Time Encodings
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_cyclical_time_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add sine/cosine encodings for `hour` and `day_of_week`."""
        logger.info("Adding cyclical time encodings for hour and day_of_week...")
        two_pi = 2.0 * math.pi
        df = df.with_columns([
            (pl.col("hour").cast(pl.Float64) * (two_pi / 24)).sin().cast(pl.Float32).alias("hour_sin"),
            (pl.col("hour").cast(pl.Float64) * (two_pi / 24)).cos().cast(pl.Float32).alias("hour_cos"),
            (pl.col("day_of_week").cast(pl.Float64) * (two_pi / 7)).sin().cast(pl.Float32).alias("dow_sin"),
            (pl.col("day_of_week").cast(pl.Float64) * (two_pi / 7)).cos().cast(pl.Float32).alias("dow_cos"),
        ])
        return df

    # ------------------------------------------------------------------ #
    # Cross Features
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_cross_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add `gender_x_cate` and `pid_x_cate` categorical cross features."""
        logger.info("Adding cross features (final_gender_code x cate_id, pid x cate_id)...")
        df = df.with_columns([
            pl.concat_str(
                [pl.col("final_gender_code").cast(pl.Utf8), pl.col("cate_id").cast(pl.Utf8)],
                separator="_",
            ).cast(pl.Categorical).alias("gender_x_cate"),
            pl.concat_str(
                [pl.col("pid").cast(pl.Utf8), pl.col("cate_id").cast(pl.Utf8)],
                separator="_",
            ).cast(pl.Categorical).alias("pid_x_cate"),
        ])
        return df

    # ------------------------------------------------------------------ #
    # Out-of-Fold Smoothed Bayesian Target Encoding
    # ------------------------------------------------------------------ #
    def _fit_te_lookup(self, fit_df: pl.DataFrame, col: str, prior: float) -> pl.DataFrame:
        """Build a {col, col_te} smoothed target-encoding lookup table from `fit_df`."""
        te_col = f"{col}_te"
        return (
            fit_df.group_by(col)
            .agg([
                pl.col("clk").sum().alias("_pos"),
                pl.col("clk").count().alias("_count"),
            ])
            .with_columns(
                ((pl.col("_pos") + self.smoothing * prior) / (pl.col("_count") + self.smoothing))
                .cast(pl.Float32)
                .alias(te_col)
            )
            .select([col, te_col])
        )

    def fit_target_encoding(self, train_df: pl.DataFrame) -> "CTRFeatureEngineer":
        """Fit smoothed target-encoding lookup tables on the full training partition."""
        logger.info(
            f"Fitting smoothed target encodings on train partition for: {self.target_encode_cols}"
        )
        self.global_ctr = float(train_df.select(pl.col("clk").mean()).item())

        self.target_encoding_maps = {
            col: self._fit_te_lookup(train_df, col, self.global_ctr)
            for col in self.target_encode_cols
        }
        return self

    def transform_target_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the train-fitted target-encoding maps; unseen categories fall back to global CTR."""
        if not self.target_encoding_maps:
            raise RuntimeError("fit_target_encoding() must be called before transform_target_encoding().")

        for col in self.target_encode_cols:
            te_col = f"{col}_te"
            df = df.join(self.target_encoding_maps[col], on=col, how="left").with_columns(
                pl.col(te_col).fill_null(self.global_ctr)
            )
        return df

    def _add_target_encoding_oof_column(
        self,
        result: pl.DataFrame,
        source: pl.DataFrame,
        col: str,
        fold_totals: pl.DataFrame,
        total_positive: int,
        total_rows: int,
    ) -> pl.DataFrame:
        """Add one OOF TE column using aggregate subtraction from ``source``."""
        te_col = f"{col}_te"
        global_stats = source.group_by(col).agg(
            [pl.col("clk").sum().alias("_global_pos"), pl.len().alias("_global_count")]
        )
        fold_stats = source.group_by([col, "_fold"]).agg(
            [pl.col("clk").sum().alias("_heldout_pos"), pl.len().alias("_heldout_count")]
        )
        denominator = pl.col("_global_count") - pl.col("_heldout_count")
        # The fold prior is fitted on every row outside the held-out fold, not
        # on the category rows outside that fold.
        prior_denominator = pl.lit(total_rows) - pl.col("_fold_count_total")
        fold_prior = pl.when(prior_denominator > 0).then(
            (pl.lit(total_positive) - pl.col("_fold_pos_total")).truediv(prior_denominator)
        ).otherwise(float(self.global_ctr or 0.0))
        return (
            result.join(global_stats, on=col, how="left")
            .join(fold_stats, on=[col, "_fold"], how="left")
            .join(fold_totals, on="_fold", how="left")
            .with_columns(
                pl.when(denominator > 0)
                .then(
                    (
                        (pl.col("_global_pos") - pl.col("_heldout_pos"))
                        + self.smoothing * fold_prior
                    ).truediv(denominator + self.smoothing)
                )
                .otherwise(fold_prior)
                .cast(pl.Float32)
                .alias(te_col)
            )
            .drop(
                [
                    "_global_pos",
                    "_global_count",
                    "_heldout_pos",
                    "_heldout_count",
                    "_fold_pos_total",
                    "_fold_count_total",
                ]
            )
        )

    def add_target_encoding_oof(self, train_df: pl.DataFrame) -> pl.DataFrame:
        """Compute out-of-fold target encodings for train so a row's label never leaks into its own encoding."""
        logger.info(f"Computing {self.n_folds}-fold OOF target encodings on train partition...")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2 for out-of-fold target encoding.")

        # Keep the established RandomState assignment for reproducibility, but
        # store only a compact fold column and never materialize n_folds filtered
        # copies of the training frame.
        rng = np.random.RandomState(self.random_seed)
        if self.n_folds <= np.iinfo(np.int8).max:
            fold_dtype, fold_polars_dtype = np.int8, pl.Int8
        elif self.n_folds <= np.iinfo(np.int16).max:
            fold_dtype, fold_polars_dtype = np.int16, pl.Int16
        else:
            fold_dtype, fold_polars_dtype = np.int32, pl.Int32
        fold_ids = rng.randint(0, self.n_folds, size=train_df.height).astype(fold_dtype)
        train_df = train_df.with_row_index("_te_row_id").with_columns(
            pl.Series("_fold", fold_ids, dtype=fold_polars_dtype)
        )
        fold_totals = train_df.group_by("_fold").agg(
            [pl.col("clk").sum().alias("_fold_pos_total"), pl.len().alias("_fold_count_total")]
        )
        total_positive = int(train_df.select(pl.col("clk").sum()).item() or 0)
        total_rows = int(train_df.height)
        result = train_df
        for col in self.target_encode_cols:
            result = self._add_target_encoding_oof_column(
                result, train_df, col, fold_totals, total_positive, total_rows
            )
        return result.sort("_te_row_id").drop(["_fold", "_te_row_id"])

    def fit_transform_partitioned(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Feature-engineer chronological partitions without concatenating all rows.

        This keeps only the current partition plus train-fitted lookup tables in
        memory.  The caller must validate that the partitions are chronological;
        exposure counters intentionally carry state from one partition to the next.
        """
        exposure_state: dict[str, pl.DataFrame] = {}

        def transform_exposure(frame: pl.DataFrame) -> pl.DataFrame:
            frame = frame.sort("time_stamp")
            for keys, name in (
                (["user", "adgroup_id"], "user_adgroup_exposure_seq"),
                (["user", "cate_id"], "user_cate_exposure_seq"),
            ):
                state = exposure_state.get(name)
                local = (pl.col("time_stamp").cum_count().over(keys) - 1).cast(pl.Int64)
                if state is None:
                    frame = frame.with_columns(local.cast(pl.Int32).alias(name))
                else:
                    state_col = f"_{name}_prior"
                    frame = (
                        frame.join(state.rename({"_count": state_col}), on=keys, how="left")
                        .with_columns(
                            (local + pl.col(state_col).fill_null(0))
                            .cast(pl.Int32)
                            .alias(name)
                        )
                        .drop(state_col)
                    )
                counts = frame.group_by(keys).agg(pl.len().alias("_count"))
                if state is None:
                    exposure_state[name] = counts
                else:
                    exposure_state[name] = (
                        pl.concat([state, counts], how="vertical_relaxed")
                        .group_by(keys)
                        .agg(pl.col("_count").sum())
                    )
            return frame

        train = transform_exposure(train_df)
        train = self.add_cross_features(self.add_cyclical_time_features(train))
        self.fit_price_stats(train)
        train = self.add_price_features(train)
        self.fit_target_encoding(train)
        train = self.add_target_encoding_oof(train)

        val = transform_exposure(val_df)
        val = self.add_price_features(
            self.add_cross_features(self.add_cyclical_time_features(val))
        )
        val = self.transform_target_encoding(val)

        test = transform_exposure(test_df)
        test = self.add_price_features(
            self.add_cross_features(self.add_cyclical_time_features(test))
        )
        test = self.transform_target_encoding(test)
        return train, val, test

    def fit_transform_partitioned_paths(
        self,
        train_path: str | Path,
        val_path: str | Path,
        test_path: str | Path,
        output_dir: str | Path,
        *,
        memory_limit: str = "12GB",
        batch_size: int = 250_000,
    ) -> Dict[str, Any]:
        """Build engineered partitions using disk-backed DuckDB execution.

        The in-memory ``fit_transform`` API remains available for local/small
        data.  This path is deliberately implemented with Arrow batches and a
        DuckDB database in the staging directory: no complete input partition,
        category lookup, exposure state, or OOF intermediate is retained as a
        Python/Polars object.  DuckDB is allowed to spill sort/hash state to
        its private temporary directory.
        """
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        paths = {"train": Path(train_path), "val": Path(val_path), "test": Path(test_path)}
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2 for out-of-fold target encoding.")

        def quote_identifier(value: str) -> str:
            return '"' + str(value).replace('"', '""') + '"'

        def quote_literal(value: str | Path) -> str:
            text = str(value).replace("\\", "/")
            return "'" + text.replace("'", "''") + "'"

        def table_columns(connection: duckdb.DuckDBPyConnection, table: str) -> List[str]:
            return [row[1] for row in connection.execute(f"DESCRIBE {quote_identifier(table)}").fetchall()]

        def write_arrow_batches(source: Path, destination: Path, split_index: int) -> int:
            """Apply cheap row-local features without collecting a partition."""
            parquet = pq.ParquetFile(source)
            writer: Optional[pq.ParquetWriter] = None
            writer_schema: Optional[pa.Schema] = None
            row_id = 0
            try:
                for batch in parquet.iter_batches(batch_size=batch_size):
                    frame = pl.from_arrow(batch)
                    frame = self.add_cross_features(self.add_cyclical_time_features(frame))
                    frame = frame.with_columns(
                        [
                            pl.lit(split_index).cast(pl.Int8).alias("_split"),
                            pl.arange(row_id, row_id + frame.height, eager=True)
                            .cast(pl.UInt64)
                            .alias("_row_id"),
                        ]
                    )
                    arrow_table = frame.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(destination, arrow_table.schema, compression="snappy")
                        writer_schema = arrow_table.schema
                    elif writer_schema is not None and arrow_table.schema != writer_schema:
                        arrow_table = arrow_table.cast(writer_schema)
                    writer.write_table(arrow_table)
                    row_id += frame.height
            finally:
                if writer is not None:
                    writer.close()
            if row_id == 0:
                raise ValueError("Feature-engineering partitions must not be empty.")
            return row_id

        def make_fold_file(connection: duckdb.DuckDBPyConnection, destination: Path, rows: int) -> None:
            rng = np.random.RandomState(self.random_seed)
            dtype = np.int8 if self.n_folds <= np.iinfo(np.int8).max else (
                np.int16 if self.n_folds <= np.iinfo(np.int16).max else np.int32
            )
            writer: Optional[pq.ParquetWriter] = None
            row_id = 0
            try:
                result = connection.execute(
                    "SELECT _row_id FROM feature_all WHERE _split=0 ORDER BY time_stamp, _row_id"
                )
                for batch in result.fetch_record_batch(rows_per_batch=batch_size):
                    size = batch.num_rows
                    folds = rng.randint(0, self.n_folds, size=size).astype(dtype, copy=False)
                    row_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
                    table = pa.table({"_row_id": row_ids, "_fold": folds})
                    if writer is None:
                        writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
                    writer.write_table(table)
                    row_id += size
            finally:
                if writer is not None:
                    writer.close()
            if row_id != rows:
                raise RuntimeError(f"Fold map row count mismatch: expected {rows}, got {row_id}.")

        def normalize_categorical_output(source: Path) -> None:
            """Restore Polars categorical logical types without loading a partition."""
            normalized = source.with_suffix(".normalized.parquet")
            parquet = pq.ParquetFile(source)
            writer: Optional[pq.ParquetWriter] = None
            writer_schema: Optional[pa.Schema] = None
            try:
                for batch in parquet.iter_batches(batch_size=batch_size):
                    frame = pl.from_arrow(batch).with_columns(
                        [
                            pl.col(column).cast(pl.Categorical)
                            for column in ("gender_x_cate", "pid_x_cate")
                            if column in batch.schema.names
                        ]
                    )
                    table = frame.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(normalized, table.schema, compression="snappy")
                        writer_schema = table.schema
                    elif writer_schema is not None and table.schema != writer_schema:
                        table = table.cast(writer_schema)
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()
            source.unlink()
            normalized.replace(source)

        temporary_root = Path(tempfile.mkdtemp(prefix=".feature_engineering_", dir=output_root))
        spill_root = temporary_root / "duckdb_tmp"
        spill_root.mkdir()
        connection: Optional[duckdb.DuckDBPyConnection] = None
        counts: Dict[str, int] = {}
        try:
            connection = duckdb.connect(str(temporary_root / "feature_engineering.duckdb"))
            connection.execute(f"SET memory_limit={quote_literal(memory_limit)}")
            connection.execute(f"SET temp_directory={quote_literal(spill_root)}")

            raw_columns = list(pq.ParquetFile(paths["train"]).schema.names)
            for index, name in enumerate(("train", "val", "test")):
                destination = temporary_root / f"{name}_base.parquet"
                counts[name] = write_arrow_batches(paths[name], destination, index)
                connection.execute(
                    f"CREATE OR REPLACE VIEW {quote_identifier(name + '_base')} AS "
                    f"SELECT * FROM read_parquet({quote_literal(destination)})"
                )

            # Validate partition chronology using scalar aggregates only.
            chronology = []
            for name in ("train", "val", "test"):
                chronology.append(connection.execute(
                    f"SELECT min(time_stamp), max(time_stamp), count(*) FROM {quote_identifier(name + '_base')}"
                ).fetchone())
            for previous, current in zip(chronology, chronology[1:]):
                if current[0] is None or previous[1] is None or current[0] <= previous[1]:
                    raise ValueError(
                        "Expected chronological partitions train -> val -> test; "
                        f"found {current[0]!r} after {previous[1]!r}."
                    )

            connection.execute(
                """CREATE OR REPLACE TABLE base_all AS
                   SELECT * FROM train_base
                   UNION ALL SELECT * FROM val_base
                   UNION ALL SELECT * FROM test_base"""
            )
            q_user, q_ad, q_cate, q_time = (quote_identifier(c) for c in ("user", "adgroup_id", "cate_id", "time_stamp"))
            connection.execute(
                f"""CREATE OR REPLACE TABLE exposure_all AS
                SELECT b.* EXCLUDE (_split, _row_id), _split, _row_id,
                    CAST(row_number() OVER (PARTITION BY {q_user}, {q_ad}
                        ORDER BY {q_time}, _split, _row_id) - 1 AS INTEGER) AS user_adgroup_exposure_seq,
                    CAST(row_number() OVER (PARTITION BY {q_user}, {q_cate}
                        ORDER BY {q_time}, _split, _row_id) - 1 AS INTEGER) AS user_cate_exposure_seq
                FROM base_all b"""
            )

            global_median = connection.execute("SELECT median(price) FROM exposure_all WHERE _split=0").fetchone()[0]
            if global_median is None:
                global_median = 0.0
            connection.execute(
                """CREATE OR REPLACE TABLE price_stats AS
                   SELECT cate_id, median(price) AS _cate_median_price
                   FROM exposure_all WHERE _split=0 GROUP BY cate_id"""
            )
            connection.execute(
                f"""CREATE OR REPLACE TABLE feature_all AS
                SELECT x.* EXCLUDE (_cate_median_price),
                    CAST(ln(1 + price) AS FLOAT) AS price_log,
                    CAST(price / CASE WHEN coalesce(_cate_median_price, {float(global_median)}) > 0
                        THEN coalesce(_cate_median_price, {float(global_median)}) ELSE 1.0 END AS FLOAT)
                        AS price_ratio_cate
                FROM (
                    SELECT e.*, p._cate_median_price
                    FROM exposure_all e
                    LEFT JOIN price_stats p ON e.cate_id IS NOT DISTINCT FROM p.cate_id
                ) x"""
            )
            self.global_median_price = float(global_median)
            self.global_ctr = float(connection.execute("SELECT avg(clk) FROM feature_all WHERE _split=0").fetchone()[0] or 0.0)
            self.cate_median_price = None
            self.target_encoding_maps = {}

            fold_path = temporary_root / "folds.parquet"
            make_fold_file(connection, fold_path, counts["train"])
            connection.execute(
                f"""CREATE OR REPLACE TABLE train_current AS
                SELECT f.*, m._fold
                FROM feature_all f JOIN read_parquet({quote_literal(fold_path)}) m USING (_row_id)
                WHERE f._split=0"""
            )
            total_positive, total_rows = connection.execute(
                "SELECT sum(clk), count(*) FROM train_current"
            ).fetchone()
            total_positive = int(total_positive or 0)
            total_rows = int(total_rows)
            connection.execute(
                """CREATE OR REPLACE TABLE fold_totals AS
                   SELECT _fold, sum(clk) AS _fold_pos_total, count(*) AS _fold_count_total
                   FROM train_current GROUP BY _fold"""
            )

            for col in self.target_encode_cols:
                qc = quote_identifier(col)
                te_col = quote_identifier(f"{col}_te")
                connection.execute(
                    f"""CREATE OR REPLACE TABLE global_stats AS
                    SELECT {qc}, sum(clk) AS _global_pos, count(*) AS _global_count
                    FROM train_current GROUP BY {qc}"""
                )
                connection.execute(
                    f"""CREATE OR REPLACE TABLE fold_stats AS
                    SELECT {qc}, _fold, sum(clk) AS _heldout_pos, count(*) AS _heldout_count
                    FROM train_current GROUP BY {qc}, _fold"""
                )
                connection.execute("DROP TABLE IF EXISTS train_next")
                connection.execute(
                    f"""CREATE TABLE train_next AS
                    SELECT t.*,
                        CAST(CASE WHEN coalesce(g._global_count, 0) - coalesce(h._heldout_count, 0) > 0
                            THEN (coalesce(g._global_pos, 0) - coalesce(h._heldout_pos, 0)
                                + {float(self.smoothing)} * CASE WHEN {total_rows} - ft._fold_count_total > 0
                                    THEN ({total_positive} - ft._fold_pos_total) / CAST({total_rows} - ft._fold_count_total AS DOUBLE)
                                    ELSE {self.global_ctr} END)
                                / (g._global_count - h._heldout_count + {float(self.smoothing)})
                            ELSE CASE WHEN {total_rows} - ft._fold_count_total > 0
                                THEN ({total_positive} - ft._fold_pos_total) / CAST({total_rows} - ft._fold_count_total AS DOUBLE)
                                ELSE {self.global_ctr} END END AS FLOAT) AS {te_col}
                    FROM train_current t
                    LEFT JOIN global_stats g ON t.{qc} IS NOT DISTINCT FROM g.{qc}
                    LEFT JOIN fold_stats h ON t.{qc} IS NOT DISTINCT FROM h.{qc} AND t._fold = h._fold
                    LEFT JOIN fold_totals ft ON t._fold = ft._fold"""
                )
                connection.execute("DROP TABLE train_current")
                connection.execute("ALTER TABLE train_next RENAME TO train_current")

            feature_columns = [
                *raw_columns,
                "user_adgroup_exposure_seq",
                "user_cate_exposure_seq",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "gender_x_cate",
                "pid_x_cate",
                "price_log",
                "price_ratio_cate",
                *[f"{col}_te" for col in self.target_encode_cols],
            ]
            select_columns = ", ".join(quote_identifier(col) for col in feature_columns)
            # Full-train lookup maps are materialized one column at a time and
            # applied to val/test sequentially; no Python lookup dictionary is retained.
            connection.execute("DROP TABLE IF EXISTS split_current")
            for split_index, name in ((1, "val"), (2, "test")):
                connection.execute(
                    f"CREATE OR REPLACE TABLE split_current AS SELECT * FROM feature_all WHERE _split={split_index}"
                )
                for col in self.target_encode_cols:
                    qc = quote_identifier(col)
                    te_col = quote_identifier(f"{col}_te")
                    connection.execute(
                        f"""CREATE OR REPLACE TABLE target_map AS
                        SELECT {qc}, CAST((sum(clk) + {float(self.smoothing)} * {self.global_ctr}) /
                            (count(*) + {float(self.smoothing)}) AS FLOAT) AS {te_col}
                        FROM train_current GROUP BY {qc}"""
                    )
                    connection.execute("DROP TABLE IF EXISTS split_next")
                    connection.execute(
                        f"""CREATE TABLE split_next AS
                        SELECT s.*,
                            CAST(coalesce(m.{te_col}, {self.global_ctr}) AS FLOAT) AS {te_col}
                        FROM split_current s
                        LEFT JOIN target_map m ON s.{qc} IS NOT DISTINCT FROM m.{qc}"""
                    )
                    connection.execute("DROP TABLE split_current")
                    connection.execute("ALTER TABLE split_next RENAME TO split_current")
                destination = temporary_root / f"{name}_fe.parquet"
                connection.execute(
                    f"COPY (SELECT {select_columns} FROM split_current ORDER BY time_stamp, _row_id) TO {quote_literal(destination)} (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
            train_destination = temporary_root / "train_fe.parquet"
            available_columns = set(table_columns(connection, "train_current"))
            missing_columns = [col for col in feature_columns if col not in available_columns]
            if missing_columns:
                raise ValueError(f"Feature schema is missing expected columns: {missing_columns}")
            connection.execute(
                f"COPY (SELECT {select_columns} FROM train_current ORDER BY time_stamp, _row_id) TO {quote_literal(train_destination)} (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            for name in ("train", "val", "test"):
                normalize_categorical_output(temporary_root / f"{name}_fe.parquet")
            for name in ("train", "val", "test"):
                source = temporary_root / f"{name}_fe.parquet"
                destination = output_root / f"{name}_fe.parquet"
                source.replace(destination)
        finally:
            if connection is not None:
                connection.close()
            shutil.rmtree(temporary_root, ignore_errors=True)

        return {
            "num_train_rows": counts["train"],
            "num_val_rows": counts["val"],
            "num_test_rows": counts["test"],
            "columns": feature_columns,
            "target_encode_columns": list(self.target_encode_cols),
            "target_encoding_smoothing": self.smoothing,
            "target_encoding_n_folds": self.n_folds,
            "target_encoding_fold_strategy": "random_state_aggregate_disk_backed",
            "global_train_ctr": self.global_ctr,
            "global_median_price": self.global_median_price,
        }

    # ------------------------------------------------------------------ #
    # Full Pipeline Orchestration
    # ------------------------------------------------------------------ #
    def fit_transform(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Run the full feature engineering pipeline across pre-split train/val/test partitions."""
        logger.info("=" * 60)
        logger.info("Starting CTR Feature Engineering Pipeline")
        logger.info("=" * 60)

        combined = pl.concat([
            train_df.with_columns(pl.lit("train").alias(_SPLIT_COL)),
            val_df.with_columns(pl.lit("val").alias(_SPLIT_COL)),
            test_df.with_columns(pl.lit("test").alias(_SPLIT_COL)),
        ])

        combined = self.add_exposure_sequence(combined)
        combined = self.add_cyclical_time_features(combined)
        combined = self.add_cross_features(combined)

        train_only = combined.filter(pl.col(_SPLIT_COL) == "train")
        self.fit_price_stats(train_only)
        combined = self.add_price_features(combined)

        train_fe = combined.filter(pl.col(_SPLIT_COL) == "train").drop(_SPLIT_COL)
        val_fe = combined.filter(pl.col(_SPLIT_COL) == "val").drop(_SPLIT_COL)
        test_fe = combined.filter(pl.col(_SPLIT_COL) == "test").drop(_SPLIT_COL)

        self.fit_target_encoding(train_fe)
        train_fe = self.add_target_encoding_oof(train_fe)
        val_fe = self.transform_target_encoding(val_fe)
        test_fe = self.transform_target_encoding(test_fe)

        logger.info("=" * 60)
        logger.info("CTR Feature Engineering Pipeline Completed Successfully!")
        logger.info("=" * 60)

        return train_fe, val_fe, test_fe
