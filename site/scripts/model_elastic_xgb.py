#!/usr/bin/env python3
# scripts/model_elastic_xgb_expanded.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# 1) Load data
fn = os.path.join("public", "data", "StudentPerformanceFactors.csv")
df = pd.read_csv(fn).dropna(subset=["Exam_Score"])

# 2) Map ordinal variables
ordinal_map = {"Low": 1, "Medium": 2, "High": 3}
df["Motivation_Num"] = df["Motivation_Level"].map(ordinal_map)
df["Resources_Num"]  = df["Access_to_Resources"].map(ordinal_map)

# 3) Define features & target
features = ["Hours_Studied", "Attendance", "Motivation_Num", "Resources_Num"]
X = df[features]
y = df["Exam_Score"]

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) ElasticNet pipeline + expanded grid search
en_pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("en", ElasticNet(max_iter=10000, random_state=42))
])

en_param_grid = {
    "en__alpha":      [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "en__l1_ratio":   [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    "en__fit_intercept": [True, False],
    "en__positive":   [False, True],        # enforce positive coefficients?
}

en_gs = GridSearchCV(
    en_pipeline,
    en_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)
en_gs.fit(X_train, y_train)

# 6) XGBoost pipeline + expanded grid search
xgb_pipeline = Pipeline([
    ("xgb", XGBRegressor(objective="reg:squarederror", random_state=42))
])

xgb_param_grid = {
    "xgb__n_estimators":     [100, 200, 300],
    "xgb__max_depth":        [3, 5, 7, 9],
    "xgb__learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "xgb__subsample":        [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.6, 0.8, 1.0],
    "xgb__gamma":            [0, 1, 5],
    "xgb__reg_alpha":        [0, 0.01, 0.1, 1],
    "xgb__reg_lambda":       [1, 10, 50]
}

xgb_gs = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)
xgb_gs.fit(X_train, y_train)

# 7) Evaluation
def report(name, gs):
    best = gs.best_estimator_
    cv_r2 = gs.best_score_
    y_pred = best.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"  Best params: {gs.best_params_}")
    print(f"  CV (5-fold)   R² = {cv_r2:.3f}")
    print(f"  Test-set      R² = {test_r2:.3f}")

report("ElasticNet", en_gs)
report("XGBoost",   xgb_gs)

# Side-by-side summary
results = pd.DataFrame([
    {
        "Model":    "ElasticNet",
        "CV_R2":    en_gs.best_score_,
        "Test_R2":  r2_score(y_test, en_gs.best_estimator_.predict(X_test))
    },
    {
        "Model":    "XGBoost",
        "CV_R2":    xgb_gs.best_score_,
        "Test_R2":  r2_score(y_test, xgb_gs.best_estimator_.predict(X_test))
    }
])
print("\nSummary:\n", results.to_string(index=False))
