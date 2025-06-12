import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt

def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def prepare_binary_target(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    """
    Binarize Severity: if >= threshold → HighSeverity = 1, else 0.
    """
    df = df.copy()
    df['HighSeverity'] = (df['Severity'] >= threshold).astype(int)
    df = df.drop(columns=['Severity'])
    return df

def train_logistic(df: pd.DataFrame, test_size=0.333, random_state=42):
    """
    Splits df → train/test, scales continuous features, fits LogisticRegression,
    and returns the trained model, scaler, plus test‐set X/y for further plotting.
    """
    # Split into X, y
    X = df.drop(columns=['HighSeverity'])
    y = df['HighSeverity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize continuous features
    continuous_cols = [
        'Sin_Hour', 'Cos_Hour', 'Sin_Dow', 'Cos_Dow',
        'Start_Lat', 'Start_Lng', 'Temperature(F)', 
        'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'
    ]
    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # Fit logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced',
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Predict on test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # Metrics
    auc_score = roc_auc_score(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)

    # Coefficients (sorted by absolute value)
    coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)

    results = {
        'model': clf,
        'scaler': scaler,
        'auc': auc_score,
        'average_precision': ap_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier,
        'coefficients': coefs,
        'X_test': X_test,
        'y_test': y_test,
        'y_proba': y_proba
    }
    return results

def plot_calibration_curve(clf, X_test, y_test, name="Logistic", save_path="figures/logistic_calibration.png"):
    """
    Plot calibration curve: true frequency vs. predicted probability.
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test, clf.predict_proba(X_test)[:, 1], n_bins=10)
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

def plot_roc_curve(y_test, y_proba, name="Logistic", save_path="figures/logistic_roc.png"):
    """
    Plot ROC curve and log AUC.
    """
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

def plot_precision_recall_curve(y_test, y_proba, name="Logistic", save_path="figures/logistic_prc.png"):
    """
    Plot Precision–Recall curve and log PR‐AUC.
    """
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
    results = train_logistic(model_df)
    print("Logistic Regression Metrics:")
    print(f"  AUC-ROC       = {results['auc']:.4f}")
    print(f"  PR-AUC        = {results['average_precision']:.4f}")
    print(f"  Precision     = {results['precision']:.3f}")
    print(f"  Recall        = {results['recall']:.3f}")
    print(f"  F1-Score      = {results['f1']:.3f}")
    print(f"  Brier Score   = {results['brier']:.4f}\n")

    print("Top 10 Logistic Coefficients (by |value|):")
    print(results['coefficients'].head(10))

    # 3) Plot Calibration
    plot_calibration_curve(
        results['model'],
        results['X_test'],
        results['y_test'],
        name="Logistic",
        save_path="figures/logistic_calibration.png"
    )
    print("Logistic calibration curve saved to 'figures/logistic_calibration.png'")

    # 4) Plot ROC Curve
    plot_roc_curve(
        results['y_test'],
        results['y_proba'],
        name="Logistic",
        save_path="figures/logistic_roc.png"
    )

    # 5) Plot Precision–Recall Curve
    plot_precision_recall_curve(
        results['y_test'],
        results['y_proba'],
        name="Logistic",
        save_path="figures/logistic_prc.png"
    )
    print("Logistic Precision–Recall curve saved to 'figures/logistic_prc.png'")
