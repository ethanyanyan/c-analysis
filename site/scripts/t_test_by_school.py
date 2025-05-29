#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats

# 1. Load & clean
df = pd.read_csv("public/data/StudentPerformanceFactors.csv")
df = df.dropna(subset=["Exam_Score", "School_Type"])

# 2. Split groups
pub = df[df.School_Type == "Public"].Exam_Score
priv = df[df.School_Type == "Private"].Exam_Score

# 3. Descriptives
def desc(series):
    return f"M={series.mean():.2f}, SD={series.std(ddof=1):.2f}, n={series.size}"
print("Public  :", desc(pub))
print("Private :", desc(priv))

# 4. t-test
tstat, pval = stats.ttest_ind(pub, priv, equal_var=True)
dfree = pub.size + priv.size - 2

# 5. Cohen's d
pooled_sd = np.sqrt(((pub.std(ddof=1)**2) + (priv.std(ddof=1)**2)) / 2)
d = (priv.mean() - pub.mean()) / pooled_sd

print(f"\nt({dfree:.0f}) = {tstat:.2f}, p = {pval:.3f}, d = {d:.2f}")
