#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_us_accidents_extended.py

Extended Feature Engineering, EDA, Statistical Testing, and Model Training
for US-Accidents Sampled Dataset (500K rows).

This version:
  • Performs preliminary EDA (temporal patterns by hour, day-of-week, month).
  • Samples a larger subset of training data (200k points).
  • Uses a lean statsmodels Logit on 10k rows (dropping zero‐variance columns) to get p-values.
  • Trains a scikit-learn Logistic Regression and an XGBoost Classifier on full 200k sample.
  • Evaluates on held-out test set (AUC, Precision, Recall, F1, Brier, calibration).
  • Saves EDA charts, coefficient tables (with p-values), and model artifacts.

Usage:
    python model_us_accidents_extended.py

Dependencies:
    - pandas
    - numpy
    - scikit-learn
    - statsmodels
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
import statsmodels.api as sm

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
RESULTS_DIR = "model_results_extended"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Configure logging
logger = logging.getLogger("USAccidentsModelExtended")
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
    for col in ["Start_Time", "End_Time", "Weather_Timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info(f"DataFrame shape: {df.shape}")
    return df

# -----------------------------------------------------------------------------
# 3) EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------

def perform_eda(df):
    """
    Generates and saves temporal-pattern charts:
      - Hourly accident frequency
      - Day-of-week accident counts
      - Monthly accident trends
    """
    logger.info("Performing EDA charts...")

    # 3.1 Hour-of-Day distribution
    df["HourOfDay"] = df["Start_Time"].dt.hour
    hour_counts = df["HourOfDay"].value_counts().sort_index()
    plt.figure(figsize=(8,4))
    plt.bar(hour_counts.index, hour_counts.values, color="skyblue", edgecolor="k")
    plt.xlabel("Hour of Day")
    plt.ylabel("Accident Count")
    plt.title("Accident Frequency by Hour of Day")
    plt.tight_layout()
    path_h = os.path.join(FIG_DIR, "eda_hour_of_day.png")
    plt.savefig(path_h, dpi=150)
    plt.close()
    logger.info(f"Saved Hour-of-Day chart: {path_h}")

    # 3.2 Day-of-Week distribution
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek  # Monday=0, Sunday=6
    dow_mapping = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    dow_counts = df["DayOfWeek"].map(dow_mapping).value_counts().reindex(
        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    )
    plt.figure(figsize=(8,4))
    plt.bar(dow_counts.index, dow_counts.values, color="salmon", edgecolor="k")
    plt.xlabel("Day of Week")
    plt.ylabel("Accident Count")
    plt.title("Accident Counts by Day of Week")
    plt.tight_layout()
    path_d = os.path.join(FIG_DIR, "eda_day_of_week.png")
    plt.savefig(path_d, dpi=150)
    plt.close()
    logger.info(f"Saved Day-of-Week chart: {path_d}")

    # 3.3 Monthly trends
    df["YearMonth"] = df["Start_Time"].dt.to_period("M")
    per_month = df.groupby("YearMonth").size()
    per_month.index = per_month.index.to_timestamp()
    plt.figure(figsize=(10,4))
    plt.plot(per_month.index, per_month.values, color="navy", linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("Accident Count")
    plt.title("Monthly Accident Counts (2016–2023 Sample)")
    plt.tight_layout()
    path_m = os.path.join(FIG_DIR, "eda_monthly_trend.png")
    plt.savefig(path_m, dpi=150)
    plt.close()
    logger.info(f"Saved Monthly Trends chart: {path_m}")

# -----------------------------------------------------------------------------
# 4) FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def engineer_features(df):
    """
    Produces a feature matrix X and target vector y for binary severity classification:
      y = 1 if Severity >=3, else 0.
    Also returns lists of POI and numeric weather columns.
    """
    logger.info("Starting feature engineering...")

    df = df.copy()
    # 4.1 TARGET
    df["HighSeverity"] = (df["Severity"] >= 3).astype(int)
    y = df["HighSeverity"].values

    # 4.2 TEMPORAL FEATURES
    df["HourOfDay"] = df["Start_Time"].dt.hour.fillna(0).astype(int)
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek.fillna(0).astype(int)
    df["MonthOfYear"] = df["Start_Time"].dt.month.fillna(1).astype(int)
    df["Sin_Hour"] = np.sin(2 * np.pi * df["HourOfDay"] / 24.0)
    df["Cos_Hour"] = np.cos(2 * np.pi * df["HourOfDay"] / 24.0)

    # 4.3 SPATIAL FIELDS
    df["Start_Lat"] = df["Start_Lat"].fillna(df["Start_Lat"].mean())
    df["Start_Lng"] = df["Start_Lng"].fillna(df["Start_Lng"].mean())

    # 4.4 ROADWAY CONTEXT FLAGS
    poi_cols = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
        "No_Exit", "Railway", "Roundabout", "Station", "Stop",
        "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
    ]
    for col in poi_cols:
        df[col] = df[col].astype(int)

    # 4.5 WEATHER FEATURES: categorize
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

    weather_num_cols = [
        "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Precipitation(in)"
    ]
    for col in weather_num_cols:
        df[col] = df[col].fillna(df[col].median())

    # 4.6 LIGHT CONDITIONS
    df["IsNight"] = (df["Sunrise_Sunset"] == "Night").astype(int)

    # 4.7 FINAL FEATURES
    feature_cols = [
        "Sin_Hour", "Cos_Hour", "DayOfWeek", "MonthOfYear",
        "Start_Lat", "Start_Lng", "IsNight"
    ] + poi_cols + weather_num_cols + ["WeatherClass", "State", "City"]

    X = df[feature_cols].copy()
    logger.info(f"Engineered features. Feature matrix shape: {X.shape}")

    return X, y, poi_cols, weather_num_cols

# -----------------------------------------------------------------------------
# 5) TRAIN/TEST SPLIT & SAMPLING
# -----------------------------------------------------------------------------

def split_and_sample(X, y, test_size=0.2, sample_size=200000, random_state=42):
    """
    Splits X, y into train/test (stratified) then samples sample_size for modeling.
    Returns:
      X_train_s, X_test, y_train_s, y_test
    """
    logger.info(f"Splitting data: test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    logger.info(f"  X_train full: {X_train.shape}, y_train full: {y_train.shape}")
    logger.info(f"  X_test:       {X_test.shape}, y_test:      {y_test.shape}")

    # Sample for modeling
    n_train = len(X_train)
    if sample_size < n_train:
        np.random.seed(random_state)
        idx = np.random.choice(np.arange(n_train), size=sample_size, replace=False)
        X_train_s = X_train.iloc[idx].reset_index(drop=True)
        y_train_s = y_train[idx]
        logger.info(f"  Sampled X_train: {X_train_s.shape}, y_train_s: {y_train_s.shape}")
    else:
        X_train_s, y_train_s = X_train, y_train
        logger.info("  Sample_size >= training size; no sampling applied.")

    return X_train_s, X_test, y_train_s, y_test

# -----------------------------------------------------------------------------
# 6) PREPROCESSOR & MODEL SETUP
# -----------------------------------------------------------------------------

def build_preprocessor(poi_cols, weather_num_cols):
    """
    Constructs a ColumnTransformer:
      - Numeric pipeline: impute + scale
      - Categorical pipeline: impute + one-hot
      - Passthrough: POI flags
    Returns preprocessor.
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
# 7) STATISTICAL TESTING VIA STATSMODELS (FASTER)
# -----------------------------------------------------------------------------

def compute_p_values(X_train_s, y_train_s, poi_cols, weather_num_cols):
    """
    Fits a lean statsmodels Logit on 10k rows (n=10k), dropping City/State,
    manually dummy-encoding WeatherClass, to compute p-values.
    Saves coefficient table with p-values to CSV and bar chart.
    """
    subsample_size = min(10000, len(X_train_s))
    idx = np.random.choice(np.arange(len(X_train_s)), size=subsample_size, replace=False)
    X_sub = X_train_s.iloc[idx].reset_index(drop=True)
    y_sub = y_train_s[idx]

    logger.info("Building design matrix for statsmodels (n=%d)...", subsample_size)

    # 7.1 Select columns: numeric + POI + WeatherClass dummies
    numeric_cols = ["Sin_Hour", "Cos_Hour", "DayOfWeek", "MonthOfYear",
                    "Start_Lat", "Start_Lng", "IsNight"]
    poi_flags = poi_cols  # already 0/1
    weather_dummies = pd.get_dummies(X_sub["WeatherClass"], prefix="Weather", drop_first=True)

    # Combine into one DataFrame
    X_design = pd.concat([
        X_sub[numeric_cols].reset_index(drop=True),
        X_sub[poi_flags].reset_index(drop=True),
        weather_dummies.reset_index(drop=True)
    ], axis=1)

    # Drop zero-variance columns
    variances = X_design.var(axis=0)
    to_keep = variances[variances > 0].index
    X_design = X_design[to_keep]

    # Add constant, then cast everything to float
    X_design = sm.add_constant(X_design).astype(float)

    logger.info("Fitting statsmodels Logit for p-values...")
    try:
        logit_model = sm.Logit(y_sub, X_design)
        result = logit_model.fit(disp=False)  # suppress output
        summary_frame = result.summary2().tables[1]

        summary_path = os.path.join(RESULTS_DIR, "logit_coefficients_pvalues.csv")
        summary_frame.to_csv(summary_path)
        logger.info(f"Saved logistic regression coefficients and p-values to: {summary_path}")

        # Plot top 15 significant coefficients by absolute value (p < 0.05)
        sig = summary_frame[summary_frame["P>|z|"] < 0.05]
        top_n = 15
        top_sig = sig.reindex(sig["Coef."].abs().sort_values(ascending=False).index).head(top_n)
        plt.figure(figsize=(8,6))
        plt.barh(top_sig.index[::-1], top_sig["Coef."][::-1], color="darkcyan", edgecolor="k")
        plt.xlabel("Coefficient Estimate")
        plt.title("Top 15 Statistically Significant Coefficients (|Coef|) — p < 0.05")
        plt.tight_layout()
        coef_plot_path = os.path.join(FIG_DIR, "stat_sig_coefficients.png")
        plt.savefig(coef_plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved significant coefficients plot: {coef_plot_path}")

    except np.linalg.LinAlgError:
        logger.warning("Statsmodels Logit failed due to singular matrix. Skipping p-values.")

# -----------------------------------------------------------------------------
# 8) TRAIN & EVALUATE MODELS
# -----------------------------------------------------------------------------

def train_and_evaluate(X_train_s, y_train_s, X_test, y_test,
                       poi_cols, weather_num_cols, preprocessor):
    """
    Trains:
      - scikit-learn Logistic Regression (on full X_train_s)
      - XGBoost Classifier (on full X_train_s)
    Evaluates on X_test/y_test.
    Saves metrics and curves to FIG_DIR.
    """
    # -------- Logistic Regression (sklearn) --------
    logger.info("Training scikit-learn Logistic Regression (C=1.0, balanced)...")
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
    t0 = datetime.now()
    lr_pipeline.fit(X_train_s, y_train_s)
    logger.info(f"Logistic Regression training time: {datetime.now() - t0}")
    evaluate_single_model(lr_pipeline, X_test, y_test, model_name="LogisticRegression")

    joblib.dump(lr_pipeline, os.path.join(RESULTS_DIR, "lr_pipeline.joblib"))
    logger.info("Saved Logistic Regression pipeline.")

    # -------- XGBoost --------
    logger.info("Training XGBoost (n_estimators=150, max_depth=6, learning_rate=0.1)...")
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4
        ))
    ])
    t1 = datetime.now()
    xgb_pipeline.fit(X_train_s, y_train_s)
    logger.info(f"XGBoost training time: {datetime.now() - t1}")
    evaluate_single_model(xgb_pipeline, X_test, y_test, model_name="XGBoost")

    joblib.dump(xgb_pipeline, os.path.join(RESULTS_DIR, "xgb_pipeline.joblib"))
    logger.info("Saved XGBoost pipeline.")

    # -------- Feature Importances (XGBoost) --------
    save_feature_importances(xgb_pipeline.named_steps["classifier"],
                             preprocessor, poi_cols, weather_num_cols)

# -----------------------------------------------------------------------------
# 9) SINGLE-MODEL EVALUATION
# -----------------------------------------------------------------------------

def evaluate_single_model(model, X_test, y_test, model_name):
    """
    Evaluates a pipeline on the test set: computes metrics and saves ROC, PR, and calibration curves.
    """
    logger.info(f"Evaluating model: {model_name}")
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
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color="navy", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(FIG_DIR, f"{model_name}_roc.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logger.info(f"Saved {model_name} ROC curve: {roc_path}")

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    plt.figure(figsize=(6,4))
    plt.plot(recall_vals, precision_vals, color="darkorange", lw=2, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(FIG_DIR, f"{model_name}_prc.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    logger.info(f"Saved {model_name} PR curve: {pr_path}")

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(6,4))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Calibration")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve: {model_name}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    cal_path = os.path.join(FIG_DIR, f"{model_name}_calibration.png")
    plt.savefig(cal_path, dpi=150)
    plt.close()
    logger.info(f"Saved {model_name} Calibration curve: {cal_path}")

# -----------------------------------------------------------------------------
# 10) FEATURE IMPORTANCES FOR XGBoost
# -----------------------------------------------------------------------------

def save_feature_importances(xgb_model, preprocessor, poi_cols, weather_num_cols):
    """
    Extracts XGBoost feature importances and saves CSV + bar plot of top 20.
    """
    logger.info("Extracting XGBoost feature importances...")

    num_cols = ["Sin_Hour", "Cos_Hour", "DayOfWeek", "MonthOfYear",
                "Start_Lat", "Start_Lng", "IsNight"] + weather_num_cols

    cat_pipeline = preprocessor.named_transformers_["cat"]
    ohe: OneHotEncoder = cat_pipeline.named_steps["onehot"]
    cat_input_cols = cat_pipeline.named_steps["imputer"].feature_names_in_
    ohe_names = ohe.get_feature_names_out(cat_input_cols)

    poi_names = poi_cols
    feature_names = list(num_cols) + list(ohe_names) + list(poi_names)
    importances = xgb_model.feature_importances_

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_csv = os.path.join(RESULTS_DIR, "xgb_feature_importances.csv")
    fi_df.to_csv(fi_csv, index=False)
    logger.info(f"Saved XGBoost feature importances to: {fi_csv}")

    top_df = fi_df.head(20).iloc[::-1]
    plt.figure(figsize=(8,6))
    plt.barh(top_df["Feature"], top_df["Importance"], color="teal", edgecolor="k")
    plt.xlabel("Relative Importance")
    plt.title("Top 20 XGBoost Feature Importances")
    plt.tight_layout()
    imp_plot = os.path.join(FIG_DIR, "xgb_top20_importances.png")
    plt.savefig(imp_plot, dpi=150)
    plt.close()
    logger.info(f"Saved XGBoost top 20 importances: {imp_plot}")

# -----------------------------------------------------------------------------
# 11) MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    # 11.1 Load data
    df = load_data(DATA_FILE)

    # 11.2 Perform EDA
    perform_eda(df)

    # 11.3 Feature engineering
    X, y, poi_cols, weather_num_cols = engineer_features(df)

    # 11.4 Split and sample (200k for modeling)
    X_train_s, X_test, y_train_s, y_test = split_and_sample(
        X, y, test_size=0.2, sample_size=200000, random_state=42
    )

    # 11.5 Compute p-values via statsmodels
    compute_p_values(X_train_s, y_train_s, poi_cols, weather_num_cols)

    # 11.6 Build preprocessor
    preprocessor = build_preprocessor(poi_cols, weather_num_cols)

    # 11.7 Train and evaluate models on full 200k sample
    train_and_evaluate(X_train_s, y_train_s, X_test, y_test,
                       poi_cols, weather_num_cols, preprocessor)

    logger.info("Extended modeling pipeline complete. All artifacts saved.")

if __name__ == "__main__":
    main()
