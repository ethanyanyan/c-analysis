# Smart Traffic Severity Prediction Pipeline

This repository contains a complete, modularized pipeline for cleaning, exploring, feature‐engineering, and modeling a 500 K‐row U.S. traffic‐accidents dataset. The goal is to predict “high‐severity” crashes (Severity ≥ 3) using both a logistic-regression baseline and an XGBoost classifier. All intermediate artifacts (cleaned data, feature‐engineered data, EDA figures, model outputs, and evaluation plots) are generated automatically.

Due to the size of the dataset, it has been excluded from the zip folder of this project. Do find the original dataset as well as the sampled dataset at [this link on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

---

## Repository Structure

```

.
├── data_preprocessing.py # Cleans the raw CSV → `cleaned_accidents.parquet`
├── eda.py # Exploratory Data Analysis → Figures 1–11 under `figures/`
├── feature_engineering.py # Builds final ML‐ready DataFrame → `model_ready_data.parquet`
├── statsmodels_pvalues.py # Computes p-values from a 10 K subsample → Figure 9 + CSV
├── logistic_regression_model.py # Trains & evaluates Logistic Regression → ROC/PR/Calibration
├── xgboost_model.py # Trains & evaluates XGBoost → ROC/PR/Calibration + importance
├── run_all.py # Orchestrates all steps in order
├── plotting_utils.py # Helper functions for EDA‐style plots
├── requirements.txt # Required Python packages
├── cleaned_accidents.parquet # (Generated) Cleaned data
├── model_ready_data.parquet # (Generated) Final feature DataFrame
└── figures/ # All generated figures (EDA 1–11, Modelling plots, etc.)
├── Figure1_missingness.png
├── Figures2-5_univariate.png
├── Figures6-8_temporal.png
├── Figure9_stat_sig_coefficients.png
├── Figure10_spatial.png
├── Figure11_crosstab.png
├── logit_coefficients_pvalues.csv
├── logistic_roc.png
├── logistic_prc.png
├── logistic_calibration.png
├── xgb_top20_importance.png
├── xgb_roc.png
├── xgb_prc.png
├── xgb_calibration.png
└── …

```

---

## Prerequisites

- **Python 3.8+**
- Recommended: create a virtual environment (venv, conda, etc.) before installing dependencies.

---

## 📥 Installation

1. **Clone the repository** (if you haven’t already):

   ```bash
   git clone <https-repo-url>
   cd smart_traffic_prediction
   ```

````

2. **Create and activate** a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   # OR
   venv\Scripts\activate         # Windows
   ```

3. **Install all required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   > **Key dependencies**:
   > `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `xgboost`, `joblib`, `matplotlib`, `seaborn` (for convenience during EDA), and `pyarrow` (for Parquet I/O).

---

## Quick Usage

You can run each step individually or invoke the orchestrator script `run_all.py` to execute all steps in sequence.

### 1) Run the entire pipeline (recommended)

```bash
python run_all.py
```

This will:

1. **Preprocess raw CSV** → produce `cleaned_accidents.parquet`
2. **Perform full EDA** → save Figures 1–11 under `figures/`
3. **Feature‐engineer** → produce `model_ready_data.parquet`
4. **Compute p-values (statsmodels)** → CSV + Figure 9
5. **Train & evaluate Logistic Regression** → metrics & save ROC/PR/Calibration plots
6. **Train & evaluate XGBoost** → metrics & save ROC/PR/Calibration + feature importances

At the end, inspect the `figures/` directory and console logs for:

- EDA charts (Figures 1–11)
- Coefficient p-values (`logit_coefficients_pvalues.csv`) + bar plot (Figure 9)
- Logistic curves: `figures/logistic_roc.png`, `figures/logistic_prc.png`, `figures/logistic_calibration.png`
- XGBoost curves: `figures/xgb_roc.png`, `figures/xgb_prc.png`, `figures/xgb_calibration.png`
- XGBoost feature-importance plot: `figures/xgb_top20_importance.png`

### 2) Run individual steps

If you want to inspect or modify a particular stage, run these scripts in order:

1. **Data Preprocessing**

   ```bash
   python data_preprocessing.py
   ```

   - **Input**: `US_Accidents_sample.csv` (CSV must be in the same folder)
   - **Output**: `cleaned_accidents.parquet`

2. **Exploratory Data Analysis (EDA)**

   ```bash
   python eda.py
   ```

   - **Input**: `cleaned_accidents.parquet`
   - **Output**: Figures 1–11 saved under `figures/`

3. **Feature Engineering**

   ```bash
   python feature_engineering.py
   ```

   - **Input**: `cleaned_accidents.parquet`
   - **Output**: `model_ready_data.parquet`

4. **Statistical Testing (p-values via statsmodels)**

   ```bash
   python statsmodels_pvalues.py
   ```

   - **Input**: `model_ready_data.parquet`
   - **Output**:

     - `figures/logit_coefficients_pvalues.csv` (coeff + p-values)
     - `figures/Figure9_stat_sig_coefficients.png` (bar chart)

5. **Logistic Regression Model**

   ```bash
   python logistic_regression_model.py
   ```

   - **Input**: `model_ready_data.parquet`
   - **Output**:

     - Console printout of metrics & top coefficients
     - `figures/logistic_roc.png`
     - `figures/logistic_prc.png`
     - `figures/logistic_calibration.png`

6. **XGBoost Model**

   ```bash
   python xgboost_model.py
   ```

   - **Input**: `model_ready_data.parquet`
   - **Output**:

     - Console printout of metrics & top‐20 feature importances
     - `figures/xgb_top20_importance.png`
     - `figures/xgb_roc.png`
     - `figures/xgb_prc.png`
     - `figures/xgb_calibration.png`

---

## Detailed Script Descriptions

### `data_preprocessing.py`

- **Loads** `US_Accidents_sample.csv` (parses date columns).
- **Drops** columns with > 60 % missingness.
- **Imputes** remaining weather & coordinate fields with medians.
- **Casts** boolean “POI” flags to integer (0/1).
- **Drops** any duplicate IDs.
- **Adds** datetime features (`Year`, `Month`, `DayOfWeek`, `Hour`, `Is_Day`).
- **Saves** cleaned DataFrame as `cleaned_accidents.parquet`.

### `eda.py`

- **Loads** `cleaned_accidents.parquet`.
- **Generates**:

  1. **Figure 1**: Top 20 columns by missing count (bar chart).
  2. **Figure 2**: Accident counts by Severity (bar).
  3. **Figure 3**: Top 10 Weather_Condition categories (horizontal bar).
  4. **Figure 4**: Temperature histogram.
  5. **Figure 5**: Visibility histogram.
  6. **Figure 6**: Accident frequency by Hour (bar).
  7. **Figure 7**: Accident count by DayOfWeek (bar).
  8. **Figure 8**: Monthly accident counts (line).
  9. **Figure 10**: Spatial heatmap (scatter2D with log‐norm) using 20 000 random samples.
  10. **Figure 11**: Severity × Weather group (clustered bar).

- **Saves** each figure under `figures/` with descriptive filenames.

> Note:
>
> - We skip “Figure 9” here (reserved for statsmodels coefficients).
> - `plotting_utils.py` contains reusable plotting functions.

### `feature_engineering.py`

- **Loads** `cleaned_accidents.parquet`.
- **Encodes**:

  - Cyclical hour (`Sin_Hour`, `Cos_Hour`) and day-of-week (`Sin_Dow`, `Cos_Dow`).
  - Binary weather flags (`Weather_Rain`, `Weather_Snow`, `Weather_Fog`, `Weather_Clear`, `Weather_Other`).
  - One-hot encodings (top-20 categories, rest labeled “Other”) for `State` and `City`.

- **Selects** relevant numeric and flag features:

  - Temporal (sin/cos), weather binaries, `Is_Day`, spatial coordinates (`Start_Lat`, `Start_Lng`), continuous weather (`Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`), plus one-hot state/city columns.

- **Drops** any rows with missing target (`Severity`) or NaNs in the chosen features.
- **Saves** the final DataFrame (features + `Severity`) as `model_ready_data.parquet`.

### `statsmodels_pvalues.py`

- **Loads** `model_ready_data.parquet`.
- **Binarizes** `HighSeverity = (Severity ≥ 3)`.
- **Randomly subsamples** 10 000 rows, resets indices (to align target/exog).
- **Drops** zero‐variance columns.
- **Fits** a statsmodels Logit (`HighSeverity ~ features`), suppressing output.
- **Saves**:

  - `figures/logit_coefficients_pvalues.csv` (coefficients + p-values).
  - **Figure 9**: Bar plot of top 15 statistically significant coefficients (|Coef|, p < 0.05) → `figures/Figure9_stat_sig_coefficients.png`.

### `logistic_regression_model.py`

- **Loads** `model_ready_data.parquet`.
- **Binarizes** `HighSeverity` (drops `Severity`).
- **Splits** 1/3 test, 2/3 train (stratified).
- **Scales** continuous features (`Sin_Hour`, `Cos_Hour`, `Sin_Dow`, `Cos_Dow`, `Start_Lat`, `Start_Lng`, `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`) via `StandardScaler`.
- **Trains** `LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")`.
- **Predicts** on the test set; computes:

  - AUC-ROC
  - Average Precision (PR‐AUC)
  - Precision, Recall, F₁ (at default threshold 0.5)
  - Brier Score

- **Extracts & prints** top 10 coefficients (by absolute value).
- **Generates & saves**:

  1. **ROC curve** → `figures/logistic_roc.png`
  2. **Precision-Recall curve** → `figures/logistic_prc.png`
  3. **Calibration curve** → `figures/logistic_calibration.png`

### `xgboost_model.py`

- **Loads** `model_ready_data.parquet`.
- **Binarizes** `HighSeverity` (drops `Severity`).
- **Splits** 1/3 test, 2/3 train (stratified).
- **Scales** same continuous features with `StandardScaler`.
- **Converts** to `xgboost.DMatrix`.
- **Trains** `xgb.train(params, dtrain, num_round)` with:

  - `objective="binary:logistic"`
  - `eval_metric="aucpr"`
  - `tree_method="hist"`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`

- **Predicts** on `dtest`; computes:

  - AUC-ROC
  - Average Precision (PR-AUC)
  - Precision, Recall, F₁ (at default threshold 0.5)
  - Brier Score

- **Extracts & prints** top 20 feature importances (gain), normalized.
- **Generates & saves**:

  1. **ROC curve** → `figures/xgb_roc.png`
  2. **Precision-Recall curve** → `figures/xgb_prc.png`
  3. **Calibration curve** → `figures/xgb_calibration.png`
  4. **Top 20 feature importances** → `figures/xgb_top20_importance.png`

---

## Interpreting the Figures

- **Figures 1–11 (EDA)**

  - Help you understand data distributions, missingness, temporal trends, spatial hotspots, and weather‐severity relationships.

- **Figure 9 (Statistical Significance)**

  - Displays the top 15 coefficients (by absolute value) whose p < 0.05 in our subsampled Logit.
  - Use this to see which features have the strongest linear association with “HighSeverity.”

- **Figures 13 & 17 (ROC Curves)**

  - Show True Positive Rate vs. False Positive Rate for thresholds in \[0, 1].
  - AUC-ROC = 0.6637 (Logistic) vs. 0.7732 (XGBoost).

- **Figures 14 & 18 (Precision-Recall Curves)**

  - Plot Precision vs. Recall across thresholds.
  - PR-AUC = 0.3389 (Logistic) vs. 0.4913 (XGBoost), which is especially meaningful on imbalanced data.

- **Figures 15 & 19 (Calibration Curves)**

  - Compare mean predicted probability to observed frequency in 10 bins.
  - Logistic: underestimates “HighSeverity” at mid-to-high probabilities.
  - XGBoost: closely follows the diagonal up to ≈ 0.7, indicating better calibration.

- **Figure 16 (XGBoost Feature Importance)**

  - A horizontal bar chart of the top 20 features by “gain” (relative contribution to model splits).
  - Confirms that roadway controls, geographic dummies, and weather features dominate.

---

## Summary

This modular project:

- **Cleans** a 500 K traffic‐accidents CSV to Parquet (`data_preprocessing.py`).
- **Explores** distributions, missingness, and spatiotemporal patterns (EDA: Figures 1–11).
- **Engineers** robust cyclical/one-hot features plus weather flags (`feature_engineering.py`).
- **Quantifies** linear‐model coefficient significance (statsmodels p-values: Figure 9).
- **Trains & evaluates** two classifiers:
  1. **Logistic Regression** (transparent, moderate recall, underestimates probabilities).
  2. **XGBoost** (stronger ranking, high precision at default threshold, better calibration).
- **Saves** all evaluation plots (ROC, PR, Calibration, Feature Importances) under `figures/`.
- **Logs** metrics to the console for easy copy-paste into reports.

Use `run_all.py` for a one-command end-to-end execution. After completion, all figures and CSVs will reside in the `figures/` directory, ready to be referenced in your final report or publication.
````
