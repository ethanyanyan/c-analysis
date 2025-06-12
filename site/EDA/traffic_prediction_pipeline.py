# traffic_prediction_pipeline.py

"""
Extended Traffic Prediction Pipeline
 - All of your original aggregation → SMOTE → RF steps, plus:
   * Exploratory figures (distribution by hour/day)
   * RF feature‐importance bar chart + partial‐dependence plots (via PartialDependenceDisplay)
   * Confusion‐matrix heatmaps + ROC curves
   * A small multinomial logistic regression
   * Export of summary tables and figures for a comprehensive write‐up

Usage:
    python traffic_prediction_pipeline.py \
        --data_path smart_mobility_dataset.csv \
        --output_dir results
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt            # for plotting
import seaborn as sns                      # for nicer heatmaps

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.inspection import PartialDependenceDisplay  # UPDATED import
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extended Traffic Prediction Pipeline with EDA + Plots + Multinomial Logit"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to smart_mobility_dataset.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save all figures and summary tables"
    )
    return parser.parse_args()

def latlng_to_grid(lat, lng, deg_per_km=0.009):
    return (int(lat / deg_per_km), int(lng / deg_per_km))

def latlng_to_grid_2km(lat, lng, deg_per_km=0.018):
    return (int(lat / deg_per_km), int(lng / deg_per_km))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    data_path = args.data_path
    out_dir = args.output_dir

    if not os.path.isfile(data_path):
        print(f"[ERROR] Dataset not found at {data_path}")
        sys.exit(1)

    # Create results directories
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "figures"))
    ensure_dir(os.path.join(out_dir, "tables"))

    # ---------------------------------------------------------------------
    # 1) Load raw data
    # ---------------------------------------------------------------------
    print("[1/15] Reading raw dataset...")
    df = pd.read_csv(data_path, parse_dates=["Timestamp"])
    print(f"     Raw dataset shape: {df.shape}")

    # (Optional) Save raw column list
    pd.DataFrame({"column": df.columns}).to_csv(
        os.path.join(out_dir, "tables", "raw_columns.csv"), index=False
    )

    # ---------------------------------------------------------------------
    # 2) Create DateOnly & Hour_Of_Day
    # ---------------------------------------------------------------------
    print("[2/15] Creating DateOnly and Hour_Of_Day columns for 1km aggregation...")
    df["DateOnly"] = df["Timestamp"].dt.date
    df["Hour_Of_Day"] = df["Timestamp"].dt.hour

    # ‼ NEW CODE: Plot distribution of raw traffic condition by hour
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="Hour_Of_Day", hue="Traffic_Condition",
                  order=list(range(24)), palette="Set2")
    plt.title("Raw Hourly Counts of Traffic_Condition (All Records)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "eda_raw_hourly_condition.png"))
    plt.close()

    # ‼ NEW CODE: Plot distribution of raw traffic condition by day of week
    df["DayOfWeek_Raw"] = df["Timestamp"].dt.day_name()
    plt.figure(figsize=(10, 5))
    order_dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.countplot(data=df, x="DayOfWeek_Raw", hue="Traffic_Condition",
                  order=order_dow, palette="Set2")
    plt.title("Raw Counts of Traffic_Condition by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "eda_raw_dow_condition.png"))
    plt.close()

    # ---------------------------------------------------------------------
    # 3) Aggregate to 1km × 1km cells
    # ---------------------------------------------------------------------
    print("[3/15] Aggregating to 1km × 1km cells...")
    df["Grid_Cell_1km"] = df.apply(
        lambda row: latlng_to_grid(row["Latitude"], row["Longitude"]), axis=1
    )
    agg_1km = df.groupby(
        ["DateOnly", "Hour_Of_Day", "Grid_Cell_1km"], as_index=False
    ).agg({
        "Vehicle_Count":            "sum",
        "Traffic_Speed_kmh":        "mean",
        "Road_Occupancy_%":         "mean",
        "Traffic_Light_State":      lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "Weather_Condition":        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "Accident_Report":          "max",
        "Sentiment_Score":          "mean",
        "Ride_Sharing_Demand":      "sum",
        "Parking_Availability":     "mean",
        "Emission_Levels_g_km":     "sum",
        "Energy_Consumption_L_h":   "mean",
        "Traffic_Condition":        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    })
    print(f"     1km aggregated shape: {agg_1km.shape}")
    print(f"     Unique 1km grid cells: {agg_1km['Grid_Cell_1km'].nunique()}")

    # Save 1km aggregation proportions
    agg_1km["Traffic_Condition"].value_counts(normalize=True).to_frame(name="prop").to_csv(
        os.path.join(out_dir, "tables", "agg1km_condition_props.csv")
    )

    # ---------------------------------------------------------------------
    # 4) Aggregate to 2km × 2km cells
    # ---------------------------------------------------------------------
    print("[4/15] Aggregating to 2km × 2km cells...")
    agg_2km = df.copy()
    agg_2km["Grid_Cell_2km"] = agg_2km.apply(
        lambda row: latlng_to_grid_2km(row["Latitude"], row["Longitude"]), axis=1
    )
    agg_2km = agg_2km.groupby(
        ["DateOnly", "Hour_Of_Day", "Grid_Cell_2km"], as_index=False
    ).agg({
        "Vehicle_Count":            "sum",
        "Traffic_Speed_kmh":        "mean",
        "Road_Occupancy_%":         "mean",
        "Traffic_Light_State":      lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "Weather_Condition":        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "Accident_Report":          "max",
        "Sentiment_Score":          "mean",
        "Ride_Sharing_Demand":      "sum",
        "Parking_Availability":     "mean",
        "Emission_Levels_g_km":     "sum",
        "Energy_Consumption_L_h":   "mean",
        "Traffic_Condition":        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    })
    print(f"     2km aggregated shape: {agg_2km.shape}")
    print(f"     Unique 2km grid cells: {agg_2km['Grid_Cell_2km'].nunique()}")

    # Save 2km aggregation proportions
    agg_2km["Traffic_Condition"].value_counts(normalize=True).to_frame(name="prop").to_csv(
        os.path.join(out_dir, "tables", "agg2km_condition_props.csv")
    )

    # ---------------------------------------------------------------------
    # 5) Drop sparse cells (< 3 observations)
    # ---------------------------------------------------------------------
    print("[5/15] Dropping 2km cells with fewer than 3 observations...")
    cell_counts_2km = agg_2km["Grid_Cell_2km"].value_counts()
    sparse_cells = cell_counts_2km[cell_counts_2km < 3].index.tolist()
    print(f"     Number of sparse cells to drop: {len(sparse_cells)}")
    agg_final = agg_2km[~agg_2km["Grid_Cell_2km"].isin(sparse_cells)].copy()
    print(f"     After dropping sparse cells, shape: {agg_final.shape}")
    print(f"     Unique 2km grid cells after drop: {agg_final['Grid_Cell_2km'].nunique()}")

    # ---------------------------------------------------------------------
    # 6) Check for missing values
    # ---------------------------------------------------------------------
    print("[6/15] Checking for missing values...")
    missing_counts = agg_final.isnull().sum()
    if missing_counts.sum() > 0:
        print("     Missing values per column:")
        print(missing_counts[missing_counts > 0])
    else:
        print("     None")

    # ---------------------------------------------------------------------
    # 7) Sort by (Grid_Cell_2km, DateTime_Hourly) & create lag‐1 features
    # ---------------------------------------------------------------------
    print("[7/15] Sorting by Grid_Cell_2km and timestamp to create lag features...")
    agg_final["DateTime_Hourly"] = pd.to_datetime(
        agg_final["DateOnly"].astype(str) + " " + agg_final["Hour_Of_Day"].astype(str) + ":00:00"
    )
    agg_final = agg_final.sort_values(["Grid_Cell_2km", "DateTime_Hourly"]).reset_index(drop=True)

    print("[8/15] Creating lag‐1 features grouped by Grid_Cell_2km...")
    grouped = agg_final.groupby("Grid_Cell_2km", group_keys=False)
    agg_final["Vehicle_Count_Lag1"] = grouped["Vehicle_Count"].shift(1)
    agg_final["Occupancy_Lag1"]     = grouped["Road_Occupancy_%"].shift(1)
    agg_final["Sentiment_Lag1"]     = grouped["Sentiment_Score"].shift(1)
    agg_final["RideShare_Lag1"]     = grouped["Ride_Sharing_Demand"].shift(1)
    agg_final["Parking_Lag1"]       = grouped["Parking_Availability"].shift(1)
    agg_final["Emissions_Lag1"]     = grouped["Emission_Levels_g_km"].shift(1)

    # ---------------------------------------------------------------------
    # 9) Drop rows with missing lag‐1 values
    # ---------------------------------------------------------------------
    print("[9/15] Dropping rows with missing lagged values...")
    pre_drop_count = len(agg_final)
    agg_lagged = agg_final.dropna(subset=[
        "Vehicle_Count_Lag1", "Occupancy_Lag1", "Sentiment_Lag1",
        "RideShare_Lag1", "Parking_Lag1", "Emissions_Lag1"
    ]).reset_index(drop=True)
    dropped = pre_drop_count - len(agg_lagged)
    print(f"     Dropped {dropped} rows due to missing lagged values. Remaining: {len(agg_lagged)}")

    # ---------------------------------------------------------------------
    # 10) Chronological 70%/15%/15% train/valid/test split
    # ---------------------------------------------------------------------
    print("[10/15] Performing chronological 70%/15%/15% train/valid/test split...")
    agg_lagged = agg_lagged.sort_values("DateTime_Hourly").reset_index(drop=True)
    N = len(agg_lagged)
    train_end = int(0.70 * N)
    valid_end = int(0.85 * N)

    train_full = agg_lagged.iloc[:train_end].copy().reset_index(drop=True)
    valid_full = agg_lagged.iloc[train_end:valid_end].copy().reset_index(drop=True)
    test_full  = agg_lagged.iloc[valid_end:].copy().reset_index(drop=True)
    print(f"     Train: {len(train_full)}, Valid: {len(valid_full)}, Test: {len(test_full)}")

    # ---------------------------------------------------------------------
    # 11) Verify Traffic_Condition proportions in each split
    # ---------------------------------------------------------------------
    print("[11/15] Verifying Traffic_Condition proportions in each split...")
    for name, subset in [("Train", train_full), ("Valid", valid_full), ("Test", test_full)]:
        props = subset["Traffic_Condition"].value_counts(normalize=True)
        print(f"     {name} Traffic_Condition proportions:\n{props}\n")
        props.to_frame(name=f"{name}_prop").to_csv(
            os.path.join(out_dir, "tables", f"{name.lower()}_condition_props.csv")
        )

    # ---------------------------------------------------------------------
    # 12) Feature Engineering (binary flags + dummies)
    # ---------------------------------------------------------------------
    print("[12/15] Feature engineering for hypothesis tests...")
    for df_ in [train_full, valid_full, test_full]:
        # 12.a) DayOfWeek dummies
        df_["DayOfWeek"] = df_["DateTime_Hourly"].dt.day_name()
        dow_dummies = pd.get_dummies(df_["DayOfWeek"], prefix="Dow", drop_first=False)
        for col in dow_dummies.columns:
            df_[col] = dow_dummies[col]

        # 12.b) HourOfDay dummies
        df_["Hour_Of_Day"] = df_["DateTime_Hourly"].dt.hour
        hod_dummies = pd.get_dummies(df_["Hour_Of_Day"], prefix="Hod", drop_first=False)
        for col in hod_dummies.columns:
            df_[col] = hod_dummies[col]

        # 12.c) Simple binary flags
        df_["Sentiment_Negative"] = (df_["Sentiment_Score"] < -0.2).astype(int)
        df_["Emission_High"]      = (df_["Emission_Levels_g_km"] >= 150).astype(int)
        df_["Snow_Flag"]          = (df_["Weather_Condition"] == "Snow").astype(int)
        df_["Rain_Flag"]          = (df_["Weather_Condition"] == "Rain").astype(int)
        df_["Fog_Flag"]           = (df_["Weather_Condition"] == "Fog").astype(int)
        df_["Clear_Flag"]         = (df_["Weather_Condition"] == "Clear").astype(int)
        df_["Weather_Adverse"]    = df_["Weather_Condition"].isin(["Rain", "Snow", "Fog"]).astype(int)

        df_["Red_Light_Flag"]    = (df_["Traffic_Light_State"] == "Red").astype(int)
        df_["Yellow_Light_Flag"] = (df_["Traffic_Light_State"] == "Yellow").astype(int)
        df_["Green_Light_Flag"]  = (df_["Traffic_Light_State"] == "Green").astype(int)

        df_["Peak_Hour"] = df_["Hour_Of_Day"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df_["Weekend"]   = df_["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)

    # 12.d) Create 75th‐percentile “High” flags (using TRAIN thresholds)
    vcount_q75    = train_full["Vehicle_Count_Lag1"].quantile(0.75)
    occup_q75     = train_full["Occupancy_Lag1"].quantile(0.75)
    rideshare_q75 = train_full["RideShare_Lag1"].quantile(0.75)
    parking_q75   = train_full["Parking_Lag1"].quantile(0.75)
    emiss_q75     = train_full["Emissions_Lag1"].quantile(0.75)

    for df_ in [train_full, valid_full, test_full]:
        df_[f"High_Vehicle_Count75"] = (df_["Vehicle_Count_Lag1"] >= vcount_q75).astype(int)
        df_[f"High_Occancy75"]       = (df_["Occupancy_Lag1"] >= occup_q75).astype(int)
        df_[f"High_RideShare75"]     = (df_["RideShare_Lag1"] >= rideshare_q75).astype(int)
        df_[f"High_Parking75"]       = (df_["Parking_Lag1"] >= parking_q75).astype(int)
        df_[f"High_Emissions75"]     = (df_["Emissions_Lag1"] >= emiss_q75).astype(int)

    # ---------------------------------------------------------------------
    # Build feature matrices for Random Forest
    # ---------------------------------------------------------------------
    def create_feature_matrix(df):
        numeric_cols = [
            "Vehicle_Count_Lag1", "Occupancy_Lag1", "Sentiment_Lag1",
            "RideShare_Lag1", "Parking_Lag1", "Emissions_Lag1",
            "Accident_Report", "Emission_High", "Weather_Adverse"
        ]
        num_feats = df[numeric_cols].copy()
        cat_feats = pd.get_dummies(df[[
            "Traffic_Light_State", "Weather_Condition", "Hour_Of_Day", "DayOfWeek"
        ]].astype(str), drop_first=True)
        X = pd.concat([num_feats.reset_index(drop=True),
                       cat_feats.reset_index(drop=True)], axis=1)
        return X

    X_train = create_feature_matrix(train_full)
    X_valid = create_feature_matrix(valid_full)
    X_test  = create_feature_matrix(test_full)

    # Reindex to ensure same columns
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Prepare target for RF
    mapping = {"Low": 0, "Medium": 1, "High": 2}
    y_train = train_full["Traffic_Condition"].map(mapping).astype(int)
    y_valid = valid_full["Traffic_Condition"].map(mapping).astype(int)
    y_test  = test_full["Traffic_Condition"].map(mapping).astype(int)

    # ---------------------------------------------------------------------
    # 13) SMOTE oversampling on training set (only minority classes)
    # ---------------------------------------------------------------------
    print("[13/15] Applying SMOTE oversampling to training set...")
    class_counts      = pd.Series(y_train).value_counts()
    majority_count    = class_counts.max()
    sampling_strategy = {0: int(majority_count), 1: int(majority_count)}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res_np, y_train_res = smote.fit_resample(X_train.values, y_train.values)
    X_train_res = pd.DataFrame(X_train_res_np, columns=X_train.columns)
    y_train_res = pd.Series(y_train_res, dtype=int)

    print("     After SMOTE, new training class distribution:")
    print(pd.Series(y_train_res).value_counts(normalize=True))

    X_train = X_train_res.copy()
    y_train = y_train_res.copy()

    # ---------------------------------------------------------------------
    # 14) Train Random Forest on resampled training data
    # ---------------------------------------------------------------------
    print("[14/15] Training Random Forest classifier on resampled data...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    print("     Random Forest training complete.")

    # ‼ NEW CODE: Plot RF feature importances (top 20)
    importances = pd.Series(rf_clf.feature_importances_, index=X_train.columns)
    importances_sorted = importances.sort_values(ascending=False).iloc[:20]

    plt.figure(figsize=(10, 6))
    importances_sorted.plot(kind="barh", color="steelblue")
    plt.gca().invert_yaxis()
    plt.title("Top‐20 Random Forest Feature Importances")
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "rf_top20_feature_importances.png"))
    plt.close()

    # ---------------------------------------------------------------------
    # 15) Validation & Test evaluation (with confusion matrices + ROC curves)
    # ---------------------------------------------------------------------
    print("[15/15] Evaluating on Validation set...")
    y_valid_pred = rf_clf.predict(X_valid)
    val_acc = accuracy_score(y_valid, y_valid_pred)
    print(f"     Validation accuracy: {val_acc:.3f}")
    print(f"     Validation confusion matrix:\n{confusion_matrix(y_valid, y_valid_pred)}")
    print("\n     Validation classification report:")
    print(classification_report(y_valid, y_valid_pred, target_names=["Low", "Medium", "High"]))

    # ‼ NEW: Save confusion matrix heatmap for VALID set
    cm_valid = confusion_matrix(y_valid, y_valid_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_valid, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Med", "High"], yticklabels=["Low", "Med", "High"])
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "confusion_matrix_valid.png"))
    plt.close()

    # ‼ NEW: ROC‐Curve on VALID (High vs. Not‐High)
    y_valid_bin = (y_valid == 2).astype(int)
    y_scores_valid = rf_clf.predict_proba(X_valid)[:, 2]
    fpr, tpr, _ = roc_curve(y_valid_bin, y_scores_valid)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="darkorange")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.title("ROC Curve (Valid) for High vs. Not High")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "roc_valid_high_vs_not.png"))
    plt.close()

    # Test set
    print("\n[Results] Test Set Performance")
    y_test_pred = rf_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"     Test accuracy: {test_acc:.3f}")
    print(f"     Test confusion matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print("\n     Test classification report:")
    print(classification_report(y_test, y_test_pred, target_names=["Low", "Medium", "High"]))

    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Med", "High"], yticklabels=["Low", "Med", "High"])
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "confusion_matrix_test.png"))
    plt.close()

    y_test_bin = (y_test == 2).astype(int)
    y_scores_test = rf_clf.predict_proba(X_test)[:, 2]
    fpr_t, tpr_t, _ = roc_curve(y_test_bin, y_scores_test)
    roc_auc_t = auc(fpr_t, tpr_t)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr_t, tpr_t, label=f"AUC = {roc_auc_t:.3f}", color="darkorange")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.title("ROC Curve (Test) for High vs. Not High")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "roc_test_high_vs_not.png"))
    plt.close()

    # ---------------------------------------------------------------------
    # 16) Partial-Dependence Plots for Top Numeric Predictors
    # ---------------------------------------------------------------------
    print("[16/] Generating partial‐dependence plots for top 2 numeric features...")
    # Identify top 2 numeric importances
    top_numeric = [
        col for col in importances_sorted.index
        if col in [
            "Vehicle_Count_Lag1",
            "Occupancy_Lag1",
            "Sentiment_Lag1",
            "RideShare_Lag1",
            "Parking_Lag1",
            "Emissions_Lag1"
        ]
    ][:2]

    if len(top_numeric) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # ‼ UPDATED: Specify target=2 for class "High"
        PartialDependenceDisplay.from_estimator(
            rf_clf,
            X_train,
            features=top_numeric,
            target=2,          # <-- ensure PDP for the "High" class in multi‐class setting
            ax=ax
        )
        fig.suptitle("Partial Dependence of ‘High’ Probability\non Top 2 Numeric Predictors")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(out_dir, "figures", "pdp_top2_numeric.png"))
        plt.close()

    # ---------------------------------------------------------------------
    # 17) Additional Multinomial Logistic Regression (Low/Medium/High)
    # ---------------------------------------------------------------------
    print("[17/] Fitting a multinomial Logistic Regression (Low/Medium/High) on 4 predictors...")

    X_multi = train_full[[
        "Vehicle_Count_Lag1", "Weather_Adverse", "Sentiment_Lag1", "RideShare_Lag1"
    ]].astype(float)

    # Standardize these predictors
    scaler = StandardScaler().fit(X_multi)
    X_multi_scaled = scaler.transform(X_multi)
    y_multi = train_full["Traffic_Condition"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)

    multi_clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=200,
        random_state=42
    )
    multi_clf.fit(X_multi_scaled, y_multi)

    # Coefficients & intercept
    coef_df = pd.DataFrame(
        multi_clf.coef_,
        index=["Low vs Rest", "Med vs Rest", "High vs Rest"],
        columns=["Vehicle_Count_Lag1","Weather_Adverse","Sentiment_Lag1","RideShare_Lag1"]
    )
    intercept_df = pd.DataFrame(
        {"Intercept": multi_clf.intercept_},
        index=["Low vs Rest","Med vs Rest","High vs Rest"]
    )

    coef_df.to_csv(os.path.join(out_dir, "tables", "multinomial_logit_coefficients.csv"))
    intercept_df.to_csv(os.path.join(out_dir, "tables", "multinomial_logit_intercepts.csv"))

    # Classification report on VALID
    X_valid_multi = valid_full[[
        "Vehicle_Count_Lag1", "Weather_Adverse", "Sentiment_Lag1", "RideShare_Lag1"
    ]]
    X_valid_multi_scaled = scaler.transform(X_valid_multi)
    y_valid_multi = valid_full["Traffic_Condition"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)
    valid_multi_pred = multi_clf.predict(X_valid_multi_scaled)
    valid_multi_acc = accuracy_score(y_valid_multi, valid_multi_pred)

    X_test_multi = test_full[[
        "Vehicle_Count_Lag1", "Weather_Adverse", "Sentiment_Lag1", "RideShare_Lag1"
    ]]
    X_test_multi_scaled = scaler.transform(X_test_multi)
    y_test_multi = test_full["Traffic_Condition"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)
    test_multi_pred = multi_clf.predict(X_test_multi_scaled)
    test_multi_acc = accuracy_score(y_test_multi, test_multi_pred)

    report_valid_multi = classification_report(
        y_valid_multi, valid_multi_pred, output_dict=True, target_names=["Low","Med","High"]
    )
    pd.DataFrame(report_valid_multi).transpose().to_csv(
        os.path.join(out_dir, "tables", "multinomial_valid_classification_report.csv")
    )

    report_test_multi = classification_report(
        y_test_multi, test_multi_pred, output_dict=True, target_names=["Low","Med","High"]
    )
    pd.DataFrame(report_test_multi).transpose().to_csv(
        os.path.join(out_dir, "tables", "multinomial_test_classification_report.csv")
    )

    print(f"     Multinomial Logit Valid Acc = {valid_multi_acc:.3f}, Test Acc = {test_multi_acc:.3f}")

    # (Optional) Scatterplot true vs. predicted for Multinomial model
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_multi, test_multi_pred, alpha=0.3, color="teal")
    plt.plot([-0.5, 2.5], [-0.5, 2.5], color="darkred", linestyle="--")
    plt.xlabel("True Class (0=Low,1=Med,2=High)")
    plt.ylabel("Predicted Class")
    plt.title("Multinomial Logit: True vs. Predicted (Test Set)")
    plt.xticks([0, 1, 2], ["Low", "Med", "High"])
    plt.yticks([0, 1, 2], ["Low", "Med", "High"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "multinomial_true_vs_pred_test.png"))
    plt.close()

    # ---------------------------------------------------------------------
    # 18) (Optional) Day‐of‐Week Logit Revisited
    # ---------------------------------------------------------------------
    print("[18/] Re‐testing Day-of-Week dummies in a single Logit (High vs. not-High)…")
    y_binary = (train_full["Traffic_Condition"] == "High").astype(int)
    dow_cols = [c for c in train_full.columns if c.startswith("Dow_") and c != "Dow_Monday"]
    X_dow = train_full[dow_cols].astype(float)
    X_dow = sm.add_constant(X_dow, has_constant="add")

    try:
        model_dow = sm.Logit(endog=y_binary, exog=X_dow).fit(disp=False)
        with open(os.path.join(out_dir, "tables", "logit_dow_summary.txt"), "w") as fh:
            fh.write(model_dow.summary().as_text())
        print("     Saved day-of-week Logit summary under tables/logit_dow_summary.txt")
    except Exception as e:
        print("!!! Day‐of‐Week Logit failed due to:", e)

    print("\n[Done] Pipeline completed successfully.")
    print(f"All figures saved under {out_dir}/figures, tables under {out_dir}/tables.")


if __name__ == "__main__":
    main()
