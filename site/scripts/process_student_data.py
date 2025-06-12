# scripts/process_student_data.py
import pandas as pd
df = pd.read_csv('public/data/StudentPerformanceFactors.csv')

# -- Numeric summaries --
numeric_cols = ['Hours_Studied','Attendance','Sleep_Hours',
                'Previous_Scores','Tutoring_Sessions',
                'Physical_Activity','Exam_Score']
num_summary = df[numeric_cols].describe().to_dict()

# -- Categorical counts --
cat_summary = {}
for col in [c for c in df.columns if c not in numeric_cols]:
    cat_summary[col] = df[col].value_counts().to_dict()

import json
with open('public/data/student_summary.json','w') as f:
    json.dump({'numeric':num_summary,'categorical':cat_summary}, f, indent=2)
