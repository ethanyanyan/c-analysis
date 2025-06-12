import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import xgboost as xgb

def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def prepare_binary_target(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    df = df.copy()
    df['HighSeverity'] = (df['Severity'] >= threshold).astype(int)
    df = df.drop(columns=['Severity'])
    return df

def train_xgb(df: pd.DataFrame, test_size=0.333, random_state=42):
    """
    Splits df → train/test, scales continuous cols, trains XGBoost via DMatrix,
    and returns trained booster, scaler, test‐set X/y, and y_proba.
    """
    X = df.drop(columns=['HighSeverity'])
    y = df['HighSeverity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale continuous variables
    continuous_cols = [
        'Sin_Hour', 'Cos_Hour', 'Sin_Dow', 'Cos_Dow',
        'Start_Lat', 'Start_Lng', 'Temperature(F)', 
        'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'
    ]
    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': random_state
    }
    num_round = 150
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=False)

    # Predictions on test set
    y_proba = bst.predict(dtest)
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    auc_score = roc_auc_score(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)

    # Feature importances (gain)
    gain_dict = bst.get_score(importance_type='gain')
    total_gain = sum(gain_dict.values()) if gain_dict else 1.0
    feature_importance = pd.DataFrame({
        'feature': list(gain_dict.keys()),
        'gain': [gain_dict[f] / total_gain for f in gain_dict.keys()]
    }).sort_values('gain', ascending=False)

    results = {
        'model': bst,
        'scaler': scaler,
        'auc': auc_score,
        'average_precision': ap_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_proba': y_proba
    }
    return results

def plot_feature_importance(feature_importance: pd.DataFrame, top_n=20, save_path="figures/xgb_top20_importance.png"):
    plt.figure(figsize=(8, 6))
    subset = feature_importance.head(top_n)
    plt.barh(subset['feature'][::-1], subset['gain'][::-1], color='seagreen')
    plt.xlabel("Relative Gain")
    plt.title(f"Top {top_n} XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_calibration_curve_xgb(bst, X_test, y_test, name="XGBoost", save_path="figures/xgb_calibration.png"):
    """
    Plot calibration curve: true frequency vs. predicted probability.
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test, bst.predict(xgb.DMatrix(X_test)), n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(f"Calibration Curve: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve_xgb(y_test, y_proba, name="XGBoost", save_path="figures/xgb_roc.png"):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='navy', lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"{name} ROC AUC = {roc_auc:.4f}")

def plot_precision_recall_curve_xgb(y_test, y_proba, name="XGBoost", save_path="figures/xgb_prc.png"):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_vals, precision_vals, color='darkorange', lw=2, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve: {name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"{name} PR‐AUC = {pr_auc:.4f}")

if __name__ == '__main__':
    os.makedirs("figures", exist_ok=True)

    # 1) Load and prepare data
    model_df = load_data('model_ready_data.parquet')
    model_df = prepare_binary_target(model_df, threshold=3)

    # 2) Train & evaluate
    results = train_xgb(model_df)
    print("XGBoost Metrics:")
    print(f"  AUC-ROC       = {results['auc']:.4f}")
    print(f"  PR-AUC        = {results['average_precision']:.4f}")
    print(f"  Precision     = {results['precision']:.3f}")
    print(f"  Recall        = {results['recall']:.3f}")
    print(f"  F1-Score      = {results['f1']:.3f}")
    print(f"  Brier Score   = {results['brier']:.4f}\n")

    print("Top 20 XGBoost Feature Importances:")
    print(results['feature_importance'].head(20))

    # 3) Plot Feature Importance
    plot_feature_importance(
        results['feature_importance'],
        top_n=20,
        save_path="figures/xgb_top20_importance.png"
    )
    print("XGBoost feature importance saved to 'figures/xgb_top20_importance.png'")

    # 4) Plot Calibration
    plot_calibration_curve_xgb(
        results['model'],
        results['X_test'],
        results['y_test'],
        name="XGBoost",
        save_path="figures/xgb_calibration.png"
    )
    print("XGBoost calibration curve saved to 'figures/xgb_calibration.png'")

    # 5) Plot ROC Curve
    plot_roc_curve_xgb(
        results['y_test'],
        results['y_proba'],
        name="XGBoost",
        save_path="figures/xgb_roc.png"
    )

    # 6) Plot Precision–Recall Curve
    plot_precision_recall_curve_xgb(
        results['y_test'],
        results['y_proba'],
        name="XGBoost",
        save_path="figures/xgb_prc.png"
    )
    print("XGBoost Precision–Recall curve saved to 'figures/xgb_prc.png'")
