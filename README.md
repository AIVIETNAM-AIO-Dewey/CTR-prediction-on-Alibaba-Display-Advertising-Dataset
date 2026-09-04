# Click-Through Rate (CTR) Prediction System
## Alibaba Display Advertising Dataset (~26.5M Impressions)

## 1. Project Overview

This project builds an end-to-end Machine Learning pipeline for Click-Through Rate (CTR) Prediction using the Alibaba Display Advertising dataset. The system predicts whether a target user will click on a specific advertisement impression (clk in {0, 1}).

CTR estimation is a fundamental component of online advertising, sponsored search, and recommendation systems. Accurate probability calibration enables optimal ad ranking, revenue maximization for publishers, and relevant experiences for users.

### Key Dataset Characteristics
- Total Interaction Logs: 26,557,961 impressions across an 8-day observation window (2017-05-06 to 2017-05-13).
- User Demographics: 1,061,768 unique registered user profiles.
- Ad Metadata: 846,811 unique ad creatives and campaign structures.
- Class Distribution: Severe class imbalance with a baseline CTR of 5.144% (1,366,056 clicks vs. 25,191,905 non-clicks).
- Core Evaluation Metrics: ROC-AUC (primary discrimination metric) and Logarithmic Loss / LogLoss (probability calibration metric).

## 2. Repository Structure

```
CTR_Prediction/
│
├── configs/
│   ├── preprocessing.yaml              # Preprocessing configurations, paths, schema types, and split dates
│   ├── feature_engineering.yaml        # Feature engineering and target encoding configuration
│   └── tree_models.yaml                # Feature scoping and CatBoost / XGBoost hyperparameters
│
├── data/
│   ├── raw/                            # Input raw CSV files (ignored by Git)
│   │   ├── raw_sample.csv              # Interaction logs (26.5M rows)
│   │   ├── user_profile.csv            # User demographic records (1.06M rows)
│   │   └── ad_feature.csv              # Ad metadata records (846K rows)
│   └── processed/                      # Preprocessed output datasets (Parquet format, ignored by Git)
│       ├── train.parquet               # Training partition (Days 1 to 6)
│       ├── val.parquet                 # Validation partition (Day 7)
│       ├── test.parquet                # Test partition (Day 8)
│       └── metadata.json               # Split statistics, baseline CTRs, and schema info
│
├── notebook/
│   ├── EDA.ipynb                       # Exploratory Data Analysis and raw data distributions
│   └── feature_analysis.ipynb          # Correlation diagnostics, Cramer's V, Mutual Information, and Decision Matrix
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/                  # Data loading, cleaning, relational merging, and time-based splitting
│   │   ├── __init__.py
│   │   ├── data_loader.py              # High-performance Polars CSV loader with sampling support
│   │   ├── cleaner.py                  # Missing value imputation and relational table joins
│   │   ├── preprocessor.py             # End-to-end preprocessing pipeline orchestrator
│   │   └── run_preprocessing.py        # CLI entry point for data processing
│   ├── features/                       # Feature engineering: exposure, price, cyclical time, cross, target encoding
│   │   ├── __init__.py
│   │   ├── feature_engineer.py         # CTRFeatureEngineer pipeline orchestrator
│   │   └── run_feature_engineering.py  # CLI entry point for feature engineering
│   ├── model/                          # Machine Learning model definitions and wrappers (Upcoming)
│   └── evaluate/                       # Model evaluation, metric calculation, and SHAP diagnostics (Upcoming)
│
├── models/                             # Serialized model artifacts (.joblib, .json)
├── experiments/                        # Experiment tracking logs and Optuna hyperparameter studies
├── outputs/                            # Evaluation plots, confusion matrices, and ROC curves
├── requirements.txt                    # Project dependencies
├── CONTRIBUTING.md                     # Team collaboration workflow and task roadmap
└── README.md                           # Project documentation
```

## 3. Project Workflow and Status

### Completed Phases
1. Exploratory Data Analysis (`notebook/EDA.ipynb`):
   - Schema verification, null rate quantification, and user/ad coverage analysis.
   - Temporal volume trends (peak hours: 20h–22h; trough hours: 03h–06h).
   - Baseline demographic and placement CTR distributions.
   - Ad fatigue discovery: CTR decreases progressively on repeated exposures to the same ad.

2. Data Preprocessing Pipeline (`src/preprocessing/`):
   - Memory-optimized data loading using Polars.
   - Robust missing value imputation for demographics (`pvalue_level`, `new_user_class_level`) and ads (`brand`, `price`).
   - Relational joining of interactions, user profiles, and ad metadata into a unified schema.
   - Temporal field extraction (`datetime`, `date`, `hour`, `day_of_week`, `is_weekend`).
   - Chronological, leak-free dataset splitting:
     - Train: 2017-05-06 to 2017-05-11 (Days 1–6)
     - Validation: 2017-05-12 (Day 7)
     - Test: 2017-05-13 (Day 8)
   - Serialization to compressed Parquet format (`data/processed/`).

3. Feature Correlation and Diagnostic Analysis (`notebook/feature_analysis.ipynb`):
   - Spearman rank correlation matrix for continuous and ordinal features.
   - Cramer's V categorical association matrix, identifying redundancy between `cms_segid` and `cms_group_id`.
   - Mutual Information (MI) and Information Value (IV) ranking against the `clk` target.
   - Formulation of the definitive Feature Decision Matrix.

4. Feature Engineering Pipeline (`src/features/`):
   - Exposure sequence counters (`user_adgroup_exposure_seq`, `user_cate_exposure_seq`) modeling ad fatigue, computed over the full chronological history.
   - Price transformations: `price_log` (`log1p(price)`) and `price_ratio_cate` (price relative to train-fitted per-`cate_id` median).
   - Cyclical time encodings: sine/cosine pairs for `hour` and `day_of_week`.
   - Cross features: `gender_x_cate` (`final_gender_code` x `cate_id`) and `pid_x_cate` (`pid` x `cate_id`).
   - Out-of-fold smoothed Bayesian target encoding for high-cardinality IDs (`cate_id`, `brand`, `customer`, `pid`), fitted exclusively on train and frozen onto val/test to prevent leakage.

5. Tree-Based Model Suite (`src/models/`):
   - `BaseCTRModel` abstract interface standardizing `fit()`, `predict_proba()`, `predict()`, `evaluate()`, `save()`, and `load()`.
   - Benchmark metrics on every partition: ROC-AUC, LogLoss, PR-AUC (Average Precision), and Brier score.
   - CatBoost wrapper: ordered boosting with native high-cardinality categorical handling via target statistics and automatic feature combinations.
   - XGBoost wrapper: histogram trees with native categorical splits (`enable_categorical`), using category dictionaries fitted on train only and frozen onto val/test.
   - Feature scoping driven by `configs/tree_models.yaml`, dropping `nonclk`, `user`, `adgroup_id`, `time_stamp`, and the collinear `cms_segid` per the Feature Decision Matrix.

### Upcoming Phases
6. Feature Selection:
   - Automated feature selection module based on Mutual Information and LightGBM Gain.

7. Remaining Model Experiments:
   - LightGBM: Fast histogram-based gradient boosting with native categorical handling.
   - RandomForest: Bagging benchmark on stratified subsets.

8. Hyperparameter Tuning, Explainability, and Ensembling:
   - Automated hyperparameter optimization using Optuna.
   - Global and local feature interpretability using SHAP.
   - Ensembling / Stacking of top-performing models for final test submission.

## 4. Quickstart Guide

### 1. Installation
Clone the repository and install required dependencies:
```bash
# Clone repository
git clone <repository_url>
cd CTR_Prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Data Preprocessing
Process raw CSV files and generate `train.parquet`, `val.parquet`, and `test.parquet`:

```bash
# Quick prototype run on 100,000 rows
python -m src.preprocessing.run_preprocessing --sample-size 100000

# 5% sample run (~1.3M rows)
python -m src.preprocessing.run_preprocessing --sample-fraction 0.05

# Full dataset run (~26.5M impressions)
python -m src.preprocessing.run_preprocessing --full
```

### 3. Running Feature Analysis
Launch Jupyter Notebook to inspect correlation heatmaps and diagnostic reports:
```bash
jupyter notebook notebook/feature_analysis.ipynb
```

### 4. Running Feature Engineering
Reads `train.parquet` / `val.parquet` / `test.parquet` from `data/processed/` and writes engineered `train_fe.parquet`, `val_fe.parquet`, `test_fe.parquet`:

```bash
python -m src.features.run_feature_engineering --config configs/feature_engineering.yaml
```

### 5. Training the Tree Models (CatBoost & XGBoost)
Reads `train_fe.parquet` / `val_fe.parquet` / `test_fe.parquet`, trains with early stopping on the
validation partition, writes artifacts to `models/` and a metric summary to
`experiments/tree_models_results.json`:

```bash
# Smoke run: both models on a 100,000-row training sample
python -m src.models.run_tree_models --use-fe --sample-size 100000

# Full engineered dataset, both models
python -m src.models.run_tree_models --use-fe --sample-size 0

# Single model on a 5% sample
python -m src.models.run_tree_models --use-fe --sample-fraction 0.05 --model catboost
python -m src.models.run_tree_models --use-fe --sample-fraction 0.05 --model xgboost
```

Hyperparameters, the feature scope, and the categorical / numeric split live in
`configs/tree_models.yaml`. Set `catboost.task_type: GPU` and `xgboost.device: cuda` there to train
on a CUDA device.

## 6. Team Contribution Guidelines

Refer to CONTRIBUTING.md for task assignments, code style rules, branch conventions, and submission workflows.
