# statsmodels_pvalues.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load_model_ready_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def compute_p_values(df: pd.DataFrame, sample_size: int = 10000, results_dir: str = "figures"):
    """
    Fits a lean statsmodels Logit on a subsample to compute p-values.
    Saves coefficient table with p-values to CSV and a bar chart of top 15 significant.
    """
    # Binarize target
    df = df.copy()
    df['HighSeverity'] = (df['Severity'] >= 3).astype(int)
    X = df.drop(columns=['Severity', 'HighSeverity'])
    y = df['HighSeverity']

    # Subsample
    n = len(X)
    subsample_size = min(sample_size, n)
    idx = np.random.choice(np.arange(n), size=subsample_size, replace=False)
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y.iloc[idx].reset_index(drop=True)   # << Reset index here to align with X_sub

    # Build design matrix: use all numeric columns in X_sub
    X_design = X_sub.copy()

    # Drop zero-variance columns
    variances = X_design.var(axis=0)
    to_keep = variances[variances > 0].index
    X_design = X_design[to_keep]

    # Add constant, cast to float
    X_design = sm.add_constant(X_design).astype(float)

    # Fit Logit
    try:
        logit_model = sm.Logit(y_sub, X_design)
        result = logit_model.fit(disp=False)
        summary_frame = result.summary2().tables[1]

        # Save summary_frame
        summary_frame.to_csv(f"{results_dir}/logit_coefficients_pvalues.csv")

        # Plot top 15 significant coefficients by absolute value (p < 0.05)
        sig = summary_frame[summary_frame["P>|z|"] < 0.05]
        top_n = 15
        top_sig = sig.reindex(sig["Coef."].abs().sort_values(ascending=False).index).head(top_n)
        plt.figure(figsize=(8,6))
        plt.barh(top_sig.index[::-1], top_sig["Coef."][::-1], color="darkcyan", edgecolor="k")
        plt.xlabel("Coefficient Estimate")
        plt.title("Figure 9: Top 15 Significant Coefficients (|Coef|) — p < 0.05")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/Figure9_stat_sig_coefficients.png", dpi=300)
        plt.close()

        print(f"Saved p-value coefficients to '{results_dir}/logit_coefficients_pvalues.csv'")
        print(f"Saved significant coefficients plot to '{results_dir}/Figure9_stat_sig_coefficients.png'")
    except np.linalg.LinAlgError:
        print("Statsmodels Logit failed due to singular matrix. Skipping p-values.")

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    df = load_model_ready_data('model_ready_data.parquet')
    compute_p_values(df, sample_size=10000, results_dir="figures")
