#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_us_accidents.py

Simplified Feature Engineering, Model Training, and Evaluation for US-Accidents Sampled Dataset.

This version reduces computational load by:
  • Sampling a subset of the training set (100k points) for model fitting.
  • Omitting extensive hyperparameter searches; using fixed, reasonable defaults.
  • Training two binary classifiers to predict “high-severity” vs. “low-severity”:
       1. Logistic Regression (with class_weight='balanced', C=1.0)
       2. XGBoost Classifier (with a small fixed parameter set)
  • Evaluating both models on the hold-out test set, computing AUC-ROC, precision, recall, F1‐score, and plotting curves.
  • Logging key steps and saving outputs to a results directory.

Usage:
    python model_us_accidents.py

Dependencies:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - matplotlib
    - joblib

Ensure that 'US_Accidents_sample.csv' is in the working directory before running.
"""

import os
import sys
import logging
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    classification_report,
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# 1) CONFIGURATION: filepaths and logger setup
# -----------------------------------------------------------------------------

DATA_FILE = "US_Accidents_sample.csv"
RESULTS_DIR = "model_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure logging
logger = logging.getLogger("USAccidentsModelSimplified")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(os.path.join(RESULTS_DIR, "modeling_log.txt"))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# -----------------------------------------------------------------------------
# 2) LOAD DATA (with datetime conversion)
# -----------------------------------------------------------------------------

def load_data(filepath):
    logger.info(f"Loading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    # Convert datetime columns
    for col in ["Start_Time", "End_Time", "Weather_Timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info(f"DataFrame shape: {df.shape}")
    return df


# -----------------------------------------------------------------------------
# 3) FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def engineer_features(df):
    """
    Produces a feature matrix X and target vector y for binary severity classification:
      y = 1 if Severity in {3,4}, else 0.
    Returns:
      X: pandas DataFrame of engineered features
      y: numpy array of binary labels
      feature_cols: list of column names in X
    """
    logger.info("Starting feature engineering...")

    df = df.copy()
    # 3.1 TARGET: Binary severity (high=1 if Severity >=3)
    df["HighSeverity"] = df["Severity"].apply(lambda x: 1 if x >= 3 else 0)
    y = df["HighSeverity"].values

    # 3.2 TEMPORAL FEATURES
    df["HourOfDay"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek   # Monday=0, Sunday=6
    df["MonthOfYear"] = df["Start_Time"].dt.month

    # Cyclical encoding for HourOfDay
    df["Sin_Hour"] = np.sin(2 * np.pi * df["HourOfDay"] / 24.0)
    df["Cos_Hour"] = np.cos(2 * np.pi * df["HourOfDay"] / 24.0)

    # 3.3 ROADWAY CONTEXT FLAGS
    poi_cols = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
        "No_Exit", "Railway", "Roundabout", "Station", "Stop",
        "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
    ]
    for col in poi_cols:
        df[col] = df[col].astype(int)

    # 3.4 ENVIRONMENTAL FEATURES: map to broad classes
    def map_weather(cond):
        if pd.isna(cond):
            return "Unknown"
        cond_lower = cond.lower()
        if "rain" in cond_lower:
            return "Rain"
        elif "snow" in cond_lower:
            return "Snow"
        elif any(tok in cond_lower for tok in ["clear", "fair"]):
            return "Clear"
        else:
            return "Other"

    df["WeatherClass"] = df["Weather_Condition"].apply(map_weather)

    # Numeric weather metrics
    weather_num_cols = [
        "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Precipitation(in)"
    ]

    # 3.5 LIGHT CONDITIONS
    df["IsNight"] = df["Sunrise_Sunset"].apply(lambda x: 1 if x == "Night" else 0)

    # 3.6 ASSEMBLE FINAL FEATURE DATAFRAME
    feature_cols = [
        "Sin_Hour", "Cos_Hour", "DayOfWeek", "MonthOfYear",
        "Start_Lat", "Start_Lng",
        "IsNight"
    ] + poi_cols + weather_num_cols + ["WeatherClass", "State", "City"]

    X = df[feature_cols].copy()
    logger.info(f"Engineered features. Feature matrix shape: {X.shape}")

    return X, y, feature_cols


# -----------------------------------------------------------------------------
# 4) TRAIN/TEST SPLIT & SAMPLING
# -----------------------------------------------------------------------------

def split_and_sample(X, y, test_size=0.2, sample_size=100000, random_state=42):
    """
    Splits X, y into training and hold-out test sets (stratified by y).
    Then randomly samples `sample_size` rows from the training set for fitting.
    Returns:
      X_train_s, X_test, y_train_s, y_test
    """
    logger.info(f"Splitting data: test_size={test_size}, stratify on y")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    logger.info(f"  X_train full: {X_train.shape}, y_train full: {y_train.shape}")
    logger.info(f"  X_test:       {X_test.shape}, y_test:      {y_test.shape}")

    # Sample a subset from the training set to reduce compute
    if sample_size < len(X_train):
        np.random.seed(random_state)
        pos_idx = np.random.choice(np.arange(len(X_train)), size=sample_size, replace=False)
        X_train_s = X_train.iloc[pos_idx].reset_index(drop=True)
        y_train_s = y_train[pos_idx]
        logger.info(f"  Sampled X_train: {X_train_s.shape}, y_train_s: {y_train_s.shape}")
    else:
        X_train_s, y_train_s = X_train, y_train
        logger.info("  Sample_size >= training size; no sampling applied.")

    return X_train_s, X_test, y_train_s, y_test


# -----------------------------------------------------------------------------
# 5) PREPROCESSOR & MODEL SETUP
# -----------------------------------------------------------------------------

def build_preprocessor(poi_cols, weather_num_cols):
    """
    Constructs a ColumnTransformer that:
      - Imputes and scales numeric columns
      - Imputes and One-Hot encodes categorical columns
      - Passes through binary POI flags as-is
    Returns:
      preprocessor: ColumnTransformer
    """
    numeric_cols = ["Sin_Hour", "Cos_Hour", "DayOfWeek", "MonthOfYear",
                    "Start_Lat", "Start_Lng", "IsNight"] + weather_num_cols
    categorical_cols = ["WeatherClass", "State", "City"]
    passthrough_cols = poi_cols

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
            ("passthrough", "passthrough", passthrough_cols),
        ],
        remainder="drop",
        sparse_threshold=0
    )

    return preprocessor


# -----------------------------------------------------------------------------
# 6) TRAIN & EVALUATE MODELS
# -----------------------------------------------------------------------------

def train_and_evaluate(X_train, y_train, X_test, y_test,
                       poi_cols, weather_num_cols):
    """
    Trains Logistic Regression and XGBoost using fixed parameters, evaluates on test set.
    Saves metrics and plots to RESULTS_DIR.
    """
    # Build preprocessing transformer
    preprocessor = build_preprocessor(poi_cols, weather_num_cols)

    # -------- Logistic Regression --------
    logger.info("Training Logistic Regression (class_weight='balanced', C=1.0)...")
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        ))
    ])

    lr_start = datetime.now()
    lr_pipeline.fit(X_train, y_train)
    logger.info(f"Logistic Regression training time: {datetime.now() - lr_start}")

    # Evaluate LR
    evaluate_single_model(
        lr_pipeline, X_test, y_test, model_name="LogisticRegression"
    )

    # Save LR model
    joblib.dump(lr_pipeline, os.path.join(RESULTS_DIR, "lr_pipeline.joblib"))
    logger.info("Saved Logistic Regression pipeline.")

    # -------- XGBoost Classifier --------
    logger.info("Training XGBoost (n_estimators=100, max_depth=6, learning_rate=0.1)...")
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4
        ))
    ])

    xgb_start = datetime.now()
    xgb_pipeline.fit(X_train, y_train)
    logger.info(f"XGBoost training time: {datetime.now() - xgb_start}")

    # Evaluate XGBoost
    evaluate_single_model(
        xgb_pipeline, X_test, y_test, model_name="XGBoost"
    )

    # Save XGBoost model
    joblib.dump(xgb_pipeline, os.path.join(RESULTS_DIR, "xgb_pipeline.joblib"))
    logger.info("Saved XGBoost pipeline.")

    # -------- Feature Importances (XGBoost) --------
    save_feature_importances(xgb_pipeline.named_steps["classifier"],
                             preprocessor, poi_cols, weather_num_cols)


def evaluate_single_model(model, X_test, y_test, model_name):
    """
    Evaluates a single trained pipeline on the test set.
    Computes AUC-ROC, Precision, Recall, F1-score, Brier Score, and plots curves.
    """
    logger.info(f"Evaluating model: {model_name} on hold-out test set")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)

    logger.info(f"{model_name} Test AUC-ROC: {auc_roc:.4f}")
    logger.info(f"{model_name} Test Precision: {precision:.4f}")
    logger.info(f"{model_name} Test Recall: {recall:.4f}")
    logger.info(f"{model_name} Test F1-score: {f1:.4f}")
    logger.info(f"{model_name} Test Brier Score: {brier:.4f}")

    report = classification_report(y_test, y_pred, target_names=["LowSeverity", "HighSeverity"])
    logger.info(f"{model_name} Classification Report:\n{report}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="navy", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, f"{model_name}_roc.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curve to: {roc_path}")

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, color="darkorange", lw=2, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(RESULTS_DIR, f"{model_name}_prc.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    logger.info(f"Saved Precision-Recall curve to: {pr_path}")

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve: {model_name}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    cal_path = os.path.join(RESULTS_DIR, f"{model_name}_calibration.png")
    plt.savefig(cal_path, dpi=150)
    plt.close()
    logger.info(f"Saved Calibration curve to: {cal_path}")


# -----------------------------------------------------------------------------
# 7) FEATURE IMPORTANCES FOR XGBoost
# -----------------------------------------------------------------------------

def save_feature_importances(xgb_model, preprocessor, poi_cols, weather_num_cols):
    """
    Extracts feature importances from the trained XGBoost model and saves a bar plot.
    """
    logger.info("Extracting feature importances from XGBoost...")

    # 7.1 Build full list of feature names after preprocessing
    # Numeric columns
    numeric_cols = (
        ["Sin_Hour", "Cos_Hour", "DayOfWeek", "MonthOfYear",
         "Start_Lat", "Start_Lng", "IsNight"] + weather_num_cols
    )

    # One-Hot encode categorical feature names
    cat_pipeline = preprocessor.named_transformers_["cat"]
    ohe: OneHotEncoder = cat_pipeline.named_steps["onehot"]
    cat_input_cols = cat_pipeline.named_steps["imputer"].feature_names_in_
    ohe_feature_names = ohe.get_feature_names_out(cat_input_cols)

    # Passthrough POI columns
    passthrough_cols = poi_cols

    all_feature_names = list(numeric_cols) + list(ohe_feature_names) + list(passthrough_cols)

    # 7.2 Retrieve importances from the XGBoost model
    importances = xgb_model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    fi_csv_path = os.path.join(RESULTS_DIR, "xgb_feature_importances.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    logger.info(f"Saved XGBoost feature importances to: {fi_csv_path}")

    # Plot top 20 features
    top_n = 20
    top_df = fi_df.head(top_n).iloc[::-1]  # reverse for horizontal bar chart
    plt.figure(figsize=(8, 6))
    plt.barh(top_df["Feature"], top_df["Importance"], color="teal", edgecolor="k")
    plt.xlabel("Relative Importance")
    plt.title(f"Top {top_n} XGBoost Feature Importances")
    plt.tight_layout()
    fi_plot_path = os.path.join(RESULTS_DIR, "xgb_top20_importances.png")
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved XGBoost top 20 importances plot to: {fi_plot_path}")


# -----------------------------------------------------------------------------
# 8) MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    # 8.1 Load data
    df = load_data(DATA_FILE)

    # 8.2 Feature engineering
    X, y, feature_cols = engineer_features(df)

    # Lists required for preprocessing
    poi_cols = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
        "No_Exit", "Railway", "Roundabout", "Station", "Stop",
        "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
    ]
    weather_num_cols = [
        "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Precipitation(in)"
    ]

    # 8.3 Split and sample
    X_train_s, X_test, y_train_s, y_test = split_and_sample(X, y,
                                                            test_size=0.2,
                                                            sample_size=100000,
                                                            random_state=42)

    # 8.4 Train and evaluate
    train_and_evaluate(X_train_s, y_train_s, X_test, y_test,
                       poi_cols, weather_num_cols)

    logger.info("Simplified modeling pipeline complete. All artifacts saved.")


if __name__ == "__main__":
    main()
