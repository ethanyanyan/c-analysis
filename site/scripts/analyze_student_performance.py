#!/usr/bin/env python3
# File: site/scripts/analyze_student_performance.py

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    # locate data relative to this script
    base = Path(__file__).resolve().parent.parent  # c-analysis/site
    data_path = base / "public" / "data" / "StudentPerformanceFactors.csv"

    # 1) load & clean
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[
        "Exam_Score", "Hours_Studied", "Attendance",
        "Motivation_Level", "Access_to_Resources",
        "Parental_Involvement"
    ])

    # 2) map the ordinal factors to numeric
    ord_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Motivation_Num"]       = df["Motivation_Level"].map(ord_map)
    df["Resources_Num"]        = df["Access_to_Resources"].map(ord_map)

    # 3) MULTIVARIABLE LINEAR REGRESSION
    X = df[["Hours_Studied", "Attendance", "Motivation_Num", "Resources_Num"]]
    X = sm.add_constant(X)
    y = df["Exam_Score"]
    model = sm.OLS(y, X).fit()

    print("=== Multiple Regression ===")
    print(model.summary())
    print("\nCoefficients:")
    print(f"  Intercept       = {model.params['const']:.3f}")
    print(f"  Hours_Studied   = {model.params['Hours_Studied']:.3f}")
    print(f"  Attendance      = {model.params['Attendance']:.3f}")
    print(f"  Motivation_Num  = {model.params['Motivation_Num']:.3f}")
    print(f"  Resources_Num   = {model.params['Resources_Num']:.3f}")
    print(f"  R-squared       = {model.rsquared:.3f}")
    print(f"  n               = {int(model.nobs)}")

    # 4) ONE-WAY ANOVA on Parental_Involvement
    print("\n=== One-way ANOVA: Exam_Score ~ Parental_Involvement ===")
    anova_mod = smf.ols("Exam_Score ~ C(Parental_Involvement)", data=df).fit()
    anova_table = sm.stats.anova_lm(anova_mod, typ=2)
    print(anova_table)

    # group-level descriptives
    grp = df.groupby("Parental_Involvement")["Exam_Score"] \
           .agg(["mean", "std", "count"])
    print("\nGroup means ± SD (n):")
    for lvl, row in grp.iterrows():
        print(f"  {lvl:6s}: {row['mean']:.2f} ± {row['std']:.2f} (n={int(row['count'])})")

    # compute eta-squared
    ss_total   = ((df["Exam_Score"] - df["Exam_Score"].mean())**2).sum()
    ss_between = anova_table.loc["C(Parental_Involvement)","sum_sq"]
    eta2 = ss_between / ss_total
    print(f"\nEta² = {eta2:.3f}")

if __name__ == "__main__":
    main()
