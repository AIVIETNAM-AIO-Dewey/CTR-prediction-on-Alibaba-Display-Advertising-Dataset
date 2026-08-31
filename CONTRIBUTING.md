# Contribution Guidelines and Development Roadmap

This document defines the collaboration workflow, coding standards, and upcoming task roadmap for all team members working on the CTR Prediction project.

## 1. Team Git Workflow

### Branching Strategy
- master / main: Production-ready, stable codebase. Direct commits to master are prohibited.
- feature/<feature-name>: For implementing new features, models, or analysis notebooks (e.g., feature/feature-engineering, feature/lightgbm-model).
- fix/<issue-name>: For bug fixes and data patching (e.g., fix/nan-handling).
- refactor/<component>: For performance improvements and code structure optimization.

### Commit Message Format
All commit messages must adhere to Conventional Commits:
```
<type>(<scope>): <short description>
```
Allowed types:
- feat: A new feature, module, or model implementation.
- fix: A bug fix or patch.
- refactor: Code restructuring without modifying external behavior.
- docs: Documentation updates (README, CONTRIBUTING, docstrings).
- test: Adding or updating test cases.
- config: Adjustments to YAML configs or requirements.

Example:
```bash
git checkout -b feature/tree-models
git commit -m "feat(model): implement LightGBM and CatBoost model wrappers with early stopping"
```

## 2. Engineering Standards

1. Memory and Performance Efficiency:
   - The raw dataset contains ~26.5M rows. Use Polars for all bulk data processing and parquet reads/writes.
   - Avoid converting full datasets to Pandas DataFrames in memory unless operating on bounded subsamples.
   - Specify explicit numeric types (pl.UInt32, pl.Int8, pl.Float32) to minimize RAM allocation.
2. Code Structure and Typing:
   - Write modular, object-oriented code with clear class interfaces.
   - Add explicit Python type annotations (typing.Union, typing.Dict, typing.Tuple, typing.Optional) to all function signatures.
   - Include standard docstrings (Google style or NumPy style) explaining arguments, return values, and behavior.
3. No Target Leakage:
   - Target encodings and scaling parameters must be fitted exclusively on the training partition (train.parquet) and then mapped onto validation (val.parquet) and test (test.parquet) sets.
   - Never include target-derived fields (nonclk, unregularized historical target stats) in the feature matrix.

## 3. Remaining Tasks and Assignment Roadmap

The following tasks constitute the remaining phases of the project:

### Task 1: Feature Engineering Module
- Location: src/features/ or src/preprocessing/feature_engineer.py
- Deliverables:
  - exposure_sequence: Cumulative count of how many times a user has encountered a specific adgroup_id or cate_id up to the current timestamp.
  - price_log: Logarithmic transformation log(1 + price) to stabilize right-skewed pricing distributions.
  - price_ratio_cate: Ratio of item price to the median price within its respective cate_id.
  - cyclical_time: Sine and cosine transformations for hour (sin(2*pi*hour/24), cos(2*pi*hour/24)) and day_of_week.
  - cross_features: High-signal interactions such as final_gender_code x cate_id and pid x cate_id.
  - target_encoding: Out-of-fold smoothed Bayesian target encoding for cate_id, brand, customer, and pid.

### Task 2: Feature Selection Engine
- Location: src/features/selector.py
- Deliverables:
  - Implementation of automated feature ranking based on Mutual Information and LightGBM Feature Importance (Gain).
  - Implementation of a feature threshold filter to drop low-importance features.
  - Integration with configs/preprocessing.yaml for feature inclusion/exclusion flags.

### Task 3: Tree-Based Model Architecture
- Location: src/model/
- Deliverables:
  - base_model.py: Abstract Base Class defining standard fit(), predict_proba(), save(), and load() methods.
  - lightgbm_model.py: LightGBM classifier wrapper utilizing histogram binning, native categorical features, and early stopping on validation LogLoss.
  - catboost_model.py: CatBoost classifier wrapper utilizing ordered target encoding and GPU/CPU acceleration.
  - xgboost_model.py: XGBoost classifier wrapper utilizing tree_method='hist' and categorical encodings.
  - random_forest_model.py: Scikit-learn RandomForest classifier wrapper for bagging baseline comparisons.

### Task 4: Evaluation and Benchmark Metrics Suite
- Location: src/evaluate/
- Deliverables:
  - metrics.py: Calculation of ROC-AUC, Logarithmic Loss (LogLoss), PR-AUC, and classification reports at varying thresholds.
  - evaluator.py: Standard evaluation pipeline to run predictions on val.parquet and test.parquet, logging metrics into structured tables.
  - plot_results.py: Automated generation of ROC curves, Precision-Recall curves, and Calibration curves saved in outputs/.

### Task 5: Automated Hyperparameter Tuning with Optuna
- Location: experiments/tune_optuna.py
- Deliverables:
  - Objective functions for Optuna optimizing validation ROC-AUC / LogLoss across LightGBM, XGBoost, and CatBoost.
  - Search spaces covering learning_rate, num_leaves, max_depth, subsample, colsample_bytree, and reg_alpha / reg_lambda.
  - Storage of Optuna study artifacts and best hyperparameter YAML configs in configs/model_configs/.

### Task 6: Model Interpretability with SHAP
- Location: src/evaluate/shap_analysis.py
- Deliverables:
  - TreeSHAP explainer integration to compute feature attributions on a representative sample of test impressions.
  - Generation and export of SHAP Summary plots, Bar plots, and Dependence plots to outputs/.

## 4. Pull Request and Code Review Checklist

Before submitting a Pull Request:
1. Ensure the code runs without runtime errors on both sample data and test partitions.
2. Confirm that no sensitive data, raw CSV files, or large Parquet datasets are tracked by Git (check .gitignore).
3. Verify that all newly created modules have docstrings and type hints.
4. Ensure evaluation metrics (ROC-AUC and LogLoss) are documented in the PR description for any model changes.
