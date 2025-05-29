#!/usr/bin/env python3
# scripts/model_comparison.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1) Load & clean
fn = os.path.join("public", "data", "StudentPerformanceFactors.csv")
df = pd.read_csv(fn).dropna(subset=["Exam_Score"])
# map ordinals
ord_map = {"Low":1,"Medium":2,"High":3}
df["Motivation_Num"   ] = df["Motivation_Level"   ].map(ord_map)
df["Resources_Num"    ] = df["Access_to_Resources"].map(ord_map)
df["Parental_Num"     ] = df["Parental_Involvement"].map(ord_map)

# 2) Features & target
X = df[["Hours_Studied","Attendance","Motivation_Num","Resources_Num","Parental_Num"]]
y = df["Exam_Score"]

# 3) train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# 4) candidate models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge(alpha=1)": Ridge(alpha=1),
    "Lasso(alpha=0.1)": Lasso(alpha=0.1),
    "RandomForest(100)": RandomForestRegressor(n_estimators=100, random_state=0),
    "SVR-RBF": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1)),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    cv_r2   = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring="r2"))
    results.append({"Model": name, "Test R2": test_r2, "CV (5-fold) R2": cv_r2})

results_df = pd.DataFrame(results).sort_values("Test R2", ascending=False)

# 5) Print & save
print("\nModel comparison:\n")
print(results_df.to_string(index=False))

out_fn = os.path.join("public","data","model_comparison_results.csv")
results_df.to_csv(out_fn, index=False)
print(f"\nSaved comparison to {out_fn}")
