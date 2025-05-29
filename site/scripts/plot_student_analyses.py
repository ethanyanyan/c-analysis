#!/usr/bin/env python3
# scripts/plot_student_analyses.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# 1) Load data
fn = os.path.join("public", "data", "StudentPerformanceFactors.csv")
df = pd.read_csv(fn)
df = df.dropna(subset=["Exam_Score"])

# 2) Create numeric versions of your ordinal factors
ordinal_map = {"Low": 1, "Medium": 2, "High": 3}
df["Motivation_Num"]     = df["Motivation_Level"].map(ordinal_map)
df["Resources_Num"]      = df["Access_to_Resources"].map(ordinal_map)
# (you can map other ordinals the same way if you wish)

# ensure images dir exists
out_dir = os.path.join("public", "images")
os.makedirs(out_dir, exist_ok=True)

# 3) Two‐sample t-test (Public vs. Private)
public  = df[df["School_Type"] == "Public"]["Exam_Score"]
private = df[df["School_Type"] == "Private"]["Exam_Score"]
t_stat, p_val = stats.ttest_ind(public, private, equal_var=True)
print(f"T-test Public vs Private: t = {t_stat:.2f}, p = {p_val:.3f}")

# 4) Simple linear regression: Exam_Score ~ Hours_Studied
simple_mod = smf.ols("Exam_Score ~ Hours_Studied", data=df).fit()
print(simple_mod.summary())

# Plot 1: scatter + regression line
plt.figure(figsize=(6,4))
plt.scatter(df["Hours_Studied"], df["Exam_Score"], s=10, alpha=0.6)
xs = np.linspace(df["Hours_Studied"].min(), df["Hours_Studied"].max(), 100)
ys = simple_mod.params.Intercept + simple_mod.params.Hours_Studied * xs
plt.plot(xs, ys, color="red", linewidth=2)
plt.xlabel("Hours Studied per Week")
plt.ylabel("Exam Score")
plt.title("Simple Regression: Exam Score vs. Hours Studied")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "simple_regression.png"), dpi=300)
plt.close()

# 5) Multiple regression: add Attendance, Motivation_Num, Resources_Num
multi_mod = smf.ols(
    "Exam_Score ~ Hours_Studied + Attendance + Motivation_Num + Resources_Num",
    data=df
).fit()
print(multi_mod.summary())

# Plot 2: residuals vs. fitted
fitted = multi_mod.fittedvalues
resid  = multi_mod.resid
plt.figure(figsize=(6,4))
plt.scatter(fitted, resid, s=10, alpha=0.6)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Multiple Regression Residuals vs. Fitted")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "multi_regression_residuals.png"), dpi=300)
plt.close()

# 6) One-way ANOVA: Exam_Score ~ Parental_Involvement
anova_mod   = smf.ols("Exam_Score ~ C(Parental_Involvement)", data=df).fit()
anova_table = sm.stats.anova_lm(anova_mod, typ=2)
print("ANOVA table:\n", anova_table)

# Plot 3: boxplot by parental involvement
plt.figure(figsize=(6,4))
df.boxplot(column="Exam_Score", by="Parental_Involvement", grid=False)
plt.suptitle("")  # remove auto suptitle
plt.xlabel("Parental Involvement")
plt.ylabel("Exam Score")
plt.title("Exam Score by Parental Involvement")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "anova_boxplot.png"), dpi=300)
plt.close()

print("All plots saved to:", out_dir)
