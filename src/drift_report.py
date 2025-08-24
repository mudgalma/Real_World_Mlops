import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load data
DATA_PATH = "data/raw/dataset.csv"
df = pd.read_csv(DATA_PATH)
mid = len(df)//2
ref, cur = df.iloc[:mid], df.iloc[mid:]

# Preprocess numeric and categorical columns (optional but recommended)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

ref[numeric_cols] = ref[numeric_cols].fillna(ref[numeric_cols].mean())
cur[numeric_cols] = cur[numeric_cols].fillna(cur[numeric_cols].mean())

ref[cat_cols] = ref[cat_cols].fillna("missing")
cur[cat_cols] = cur[cat_cols].fillna("missing")

from sklearn.preprocessing import OrdinalEncoder
if cat_cols:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ref[cat_cols] = enc.fit_transform(ref[cat_cols])
    cur[cat_cols] = enc.transform(cur[cat_cols])

# Create report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("drift_report.html")
print("Saved drift_report.html")

