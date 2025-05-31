#!/usr/bin/env python3
# scripts/regression_scan.py

import os
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import linregress
import argparse

# ─── 1) Load & filter ──────────────────────────────────────────────
DATA_PATH = os.path.join("public", "data", "StudentPerformanceFactors.csv")
df = pd.read_csv(DATA_PATH).dropna(subset=["Exam_Score"])

# ─── 2) Map your ordinal string columns ────────────────────────────
ord3 = {"Low": 1, "Medium": 2, "High": 3}
df["Family_Income_Num"]        = df["Family_Income"].map(ord3)
df["Distance_from_Home_Num"]   = df["Distance_from_Home"].map({"Near":1,"Moderate":2,"Far":3})
df["Motivation_Level_Num"]     = df["Motivation_Level"].map(ord3)
df["Parental_Involvement_Num"] = df["Parental_Involvement"].map(ord3)
df["Access_to_Resources_Num"]  = df["Access_to_Resources"].map(ord3)

# ─── 3) Select only numeric + mapped ordinal columns ───────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

def scan_all_pairs(df, numeric_cols):
    results = []
    for x_key, y_key in combinations(numeric_cols, 2):
        sub = df[[x_key, y_key]].dropna()
        if len(sub) < 3:
            continue
        slope, intercept, r_val, p_val, _ = linregress(sub[x_key], sub[y_key])
        results.append({
            "x": x_key,
            "y": y_key,
            "n": len(sub),
            "slope": slope,
            "intercept": intercept,
            "r2": r_val**2,
            "p_value": p_val
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan all numeric/ordinal column-pairs for linear-regression stats."
    )
    parser.add_argument(
        "--out", "-o",
        help="If set, write full results to this CSV file",
        default=None
    )
    args = parser.parse_args()

    print(f"Scanning {len(numeric_cols)} numeric/ordinal columns → {len(numeric_cols)*(len(numeric_cols)-1)//2} pairs…")
    res_df = scan_all_pairs(df, numeric_cols)
    res_df = res_df.sort_values("p_value")

    # print top 20
    print("\nTop 20 most significant pairs\n")
    print(res_df.head(20).to_string(index=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        res_df.to_csv(args.out, index=False)
        print(f"\nFull results saved to {args.out}")
