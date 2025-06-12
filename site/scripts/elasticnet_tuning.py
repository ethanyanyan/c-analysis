#!/usr/bin/env python3
# scripts/elasticnet_tuning.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import joblib

# 1) Load & filter
DATA_PATH = os.path.join("public", "data", "StudentPerformanceFactors.csv")
df = pd.read_csv(DATA_PATH).dropna(subset=["Exam_Score"])

# 2) Only keep rows that have values in all the predictors we care about
predictors = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Family_Income",        # ordinal string
    "Distance_from_Home",   # ordinal string
    "Motivation_Level",     # ordinal string
    "Parental_Involvement", # ordinal string
    "Access_to_Resources",  # ordinal string
    "Physical_Activity",    # numeric hours/week
    "Peer_Influence",       # pure category
    "Internet_Access",      # pure category
    "Learning_Disabilities",# pure category
    "Gender",               # pure category
]
df = df.dropna(subset=predictors)

# 3) Map only the string‐ordinal columns to numbers
ord3 = {"Low":1, "Medium":2, "High":3}
df["Family_Income_Num"]        = df["Family_Income"].map(ord3)
df["Distance_from_Home_Num"]   = df["Distance_from_Home"].map({"Near":1,"Moderate":2,"Far":3})
df["Motivation_Level_Num"]     = df["Motivation_Level"].map(ord3)
df["Parental_Involvement_Num"] = df["Parental_Involvement"].map(ord3)
df["Access_to_Resources_Num"]  = df["Access_to_Resources"].map(ord3)

# 4) Define numeric vs categorical
numeric_features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",       # ORIGINAL numeric
    "Family_Income_Num",
    "Distance_from_Home_Num",
    "Motivation_Level_Num",
    "Parental_Involvement_Num",
    "Access_to_Resources_Num",
]
categorical_features = [
    "Peer_Influence",
    "Internet_Access",
    "Learning_Disabilities",
    "Gender",
]

# 5) Split X / y
X = df[numeric_features + categorical_features]
y = df["Exam_Score"]

# 6) Build preprocessor: scale numeric, one‐hot the rest
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
    ]
)

# 7) Pipeline: preprocess → ElasticNet
pipeline = Pipeline([
    ("pre", preprocessor),
    ("en",  ElasticNet(max_iter=5000, random_state=42)),
])

# 8) Hyperparameter grid
param_grid = {
    "en__alpha":    [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "en__l1_ratio": [0.5, 0.7, 0.9, 1.0],
}

# 9) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10) Grid‐search (5‐fold CV, R²)
gs = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2,
    error_score="raise",
)
gs.fit(X_train, y_train)

# 11) Evaluate
best_model = gs.best_estimator_
cv_r2      = gs.best_score_
test_r2    = r2_score(y_test, best_model.predict(X_test))

print("\nElasticNet Tuning Results")
print(f"  Best α        = {gs.best_params_['en__alpha']}")
print(f"  Best l1_ratio = {gs.best_params_['en__l1_ratio']}")
print(f"  CV (5-fold)   R² = {cv_r2:.4f}")
print(f"  Test-set      R² = {test_r2:.4f}")

# 12) Save the entire pipeline (preprocessor + model) for frontend use
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/elasticnet_best.joblib")
print("Saved tuned pipeline to models/elasticnet_best.joblib")
