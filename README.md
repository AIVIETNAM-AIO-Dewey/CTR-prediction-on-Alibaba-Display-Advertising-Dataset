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
│   └── preprocessing.yaml              # Preprocessing configurations, paths, schema types, and split dates
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
│   ├── data/                           # Automated dataset downloading and verification
│   │   ├── __init__.py
│   │   └── data_downloader.py          # Kaggle dataset downloader with CLI & credential check
│   ├── preprocessing/                  # Data loading, cleaning, relational merging, and time-based splitting
│   │   ├── __init__.py
│   │   ├── data_loader.py              # High-performance Polars CSV loader with sampling support
│   │   ├── cleaner.py                  # Missing value imputation and relational table joins
│   │   ├── preprocessor.py             # End-to-end preprocessing pipeline orchestrator
│   │   └── run_preprocessing.py        # CLI entry point for data processing
│   ├── features/                       # Feature engineering and transformation pipeline
│   │   ├── feature_engineering.py      # Core feature engineering implementations in Polars
│   │   └── run_feature_engineering.py  # CLI entry point for generating engineered partitions
│   └── models/                         # Standardized baseline and ML model wrappers
│       ├── __init__.py
│       ├── base_model.py               # Abstract base model class with evaluation & serialization
│       ├── data_utils.py               # Polars dataset loader and batch sampling utilities
│       ├── logistic_regression_model.py # Logistic Regression baseline with One-Hot encoding & scaling
│       ├── lightgbm_model.py           # LightGBM GBDT model with early stopping & categorical handling
│       └── run_baseline.py             # Unified CLI entry point to train, evaluate, and compare models
│
├── models/                             # Serialized model artifacts (.joblib, .json)
├── experiments/                        # Experiment tracking logs and benchmark metrics
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
   - Exposure sequence counters (ad fatigue modeling).
   - Log-price transformations and category-relative price ratios.
   - Cyclical temporal transformations (`sin`/`cos` hour).
   - Out-of-fold Bayesian smoothed target encodings for high-cardinality IDs (`cate_id`, `brand`, `customer`, `pid`).
   - Output parquet generation (`train_fe.parquet`, `val_fe.parquet`, `test_fe.parquet`).

5. Baseline Model Benchmarking (`src/models/`):
   - Standardized OOP wrappers with uniform `fit()`, `predict_proba()`, `evaluate()`, and `save()`/`load()` interfaces.
   - **Logistic Regression**: Scaled continuous features + One-Hot Encoded categoricals.
   - **LightGBM**: Fast histogram-based GBDT with native Polars integration, early stopping, and metric tracking (ROC-AUC, LogLoss).

### Upcoming Phases
6. Multi-Model Expansion & Hyperparameter Optimization:
   - CatBoost and XGBoost implementations.
   - Automated hyperparameter optimization using Optuna.
   - Global and local feature interpretability using SHAP.
   - Stacking / Ensembling of top-performing models for final submission.

## 4. Quickstart Guide

### 1. Installation
Clone the repository and install required dependencies:
```bash
# Clone repository
git clone <repository_url>
cd CTR_Prediction

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Raw Dataset
Download and extract the Alibaba Display Advertising Dataset automatically using `kagglehub`:
```bash
python -m src.data.data_downloader
```

### 3. Running Data Preprocessing
Process raw CSV files and generate `train.parquet`, `val.parquet`, and `test.parquet`:

```bash
# Quick prototype run on 100,000 rows
python -m src.preprocessing.run_preprocessing --sample-size 100000

# 5% sample run (~1.3M rows)
python -m src.preprocessing.run_preprocessing --sample-fraction 0.05

# Full dataset run (~26.5M impressions)
python -m src.preprocessing.run_preprocessing --full
```

### 4. Running Feature Engineering
Apply feature engineering transformations (exposure counters, price ratios, out-of-fold target encoding) to generate `train_fe.parquet`, `val_fe.parquet`, and `test_fe.parquet`:

```bash
python -m src.features.run_feature_engineering
```

### 5. Running Model Training & Evaluation
Train and evaluate baseline models (Logistic Regression and LightGBM) on the preprocessed or feature-engineered datasets:

#### **A. Train All Models**
```bash
# Quick prototype run (100,000 rows)
python -m src.models.run_baseline --model all --use-fe --sample-size 100000

# 5% sample run (~1.3M rows)
python -m src.models.run_baseline --model all --use-fe --sample-fraction 0.05

# Full dataset run
python -m src.models.run_baseline --model all --use-fe --sample-size 0
```

#### **B. Train Individual Models**
```bash
# Logistic Regression only
python -m src.models.run_baseline --model lr --use-fe --sample-size 100000

# LightGBM only
python -m src.models.run_baseline --model lightgbm --use-fe --sample-size 100000
```

#### **CLI Parameters Reference**
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | `all` | Model to run: `all`, `lr`, or `lightgbm` |
| `--use-fe` | `flag` | `False` | Use feature-engineered partitions (`train_fe.parquet`) |
| `--sample-size` | `int` | `100000` | Number of training rows to sample (`0` for full dataset) |
| `--sample-fraction` | `float` | `None` | Sample fraction (e.g. `0.05` for 5%) |
| `--processed-dir` | `str` | `data/processed` | Directory containing parquet dataset partitions |
| `--models-dir` | `str` | `models` | Directory to save trained `.joblib` model artifacts |
| `--output-metrics` | `str` | `experiments/baseline_results.json` | Destination path for evaluation metrics summary |
| `--seed` | `int` | `42` | Random seed for sampling and reproducibility |

### 6. Running Feature Analysis
Launch Jupyter Notebook to inspect correlation heatmaps and diagnostic reports:
```bash
jupyter notebook notebook/feature_analysis.ipynb
```

## 5. Team Contribution Guidelines

Refer to CONTRIBUTING.md for task assignments, code style rules, branch conventions, and submission workflows.

