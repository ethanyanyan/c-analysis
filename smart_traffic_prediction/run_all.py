# run_all.py

import os
import sys
import subprocess

# 1) Ensure the “figures” directory exists
os.makedirs("figures", exist_ok=True)

# 2) Run data_preprocessing.py → creates cleaned_accidents.parquet
print("\n=== 1) Data Preprocessing ===")
subprocess.run([sys.executable, "data_preprocessing.py"], check=True)

# 3) Run eda.py → saves Figures 1–11 under figures/
print("\n=== 2) Exploratory Data Analysis (EDA) ===")
subprocess.run([sys.executable, "eda.py"], check=True)

# 4) Run feature_engineering.py → creates model_ready_data.parquet
print("\n=== 3) Feature Engineering ===")
subprocess.run([sys.executable, "feature_engineering.py"], check=True)

# 5) Run statsmodels_pvalues.py → saves coefficient p-values and Figure 9
print("\n=== 4) Statistical Testing via Statsmodels ===")
subprocess.run([sys.executable, "statsmodels_pvalues.py"], check=True)

# 6) Run logistic_regression_model.py → prints metrics, saves ROC, PR, calibration
print("\n=== 5) Train & Evaluate Logistic Regression ===")
subprocess.run([sys.executable, "logistic_regression_model.py"], check=True)

# 7) Run xgboost_model.py → prints metrics, saves feature importance, ROC, PR, calibration
print("\n=== 6) Train & Evaluate XGBoost Model ===")
subprocess.run([sys.executable, "xgboost_model.py"], check=True)

print("\nAll steps completed. Check the `figures/` folder and console output for details.")
