# scripts/data_validation.py
import pandas as pd
import json
import os
import sys
from ydata_profiling import ProfileReport

RAW = "data/raw/dataset.csv"
REPORT = "reports/data_validation.html"
SCHEMA = "schemas/heart_schema.json"
METRICS = "reports/data_metrics.json"  # small json for drift checks

os.makedirs("reports", exist_ok=True)
os.makedirs("schemas", exist_ok=True)

if not os.path.exists(RAW):
    print("ERROR: raw data not found:", RAW)
    sys.exit(2)

df = pd.read_csv(RAW)

# 1) Generate human-readable profile (detailed)
profile = ProfileReport(df, title="Heart Disease Data Validation Report", explorative=True)
profile.to_file(REPORT)
print("Wrote validation report:", REPORT)

# 2) Save small metrics used by drift detection
metrics = {
    "n_rows": int(len(df)),
    "columns": df.columns.tolist(),
    "dtypes": {c: str(df[c].dtype) for c in df.columns},
    "summary": {c: {"mean": None, "min": None, "max": None} for c in df.select_dtypes(include=["number"]).columns}
}
for c in metrics["summary"].keys():
    metrics["summary"][c]["mean"] = float(df[c].mean())
    metrics["summary"][c]["min"] = float(df[c].min())
    metrics["summary"][c]["max"] = float(df[c].max())

with open(METRICS, "w") as f:
    json.dump(metrics, f, indent=2)
print("Wrote numeric metrics for drift:", METRICS)

# 3) Schema check - create baseline if missing; else compare
if not os.path.exists(SCHEMA):
    print("Saving baseline schema and metrics (first run).")
    with open(SCHEMA, "w") as f:
        json.dump({"columns": metrics["columns"], "dtypes": metrics["dtypes"]}, f, indent=2)
    print("Baseline schema saved to", SCHEMA)
    sys.exit(0)

with open(SCHEMA) as f:
    baseline = json.load(f)

if baseline["columns"] != metrics["columns"]:
    print("SCHEMA MISMATCH! Expected columns:", baseline["columns"])
    print("Found columns:", metrics["columns"])
    sys.exit(1)

# Extra checks: missing values threshold for each column (example)
max_missing_pct = 0.2
for c in df.columns:
    miss_pct = df[c].isna().mean()
    if miss_pct > max_missing_pct:
        print(f"TOO MANY MISSING VALUES: column={c} missing_pct={miss_pct:.2f} > {max_missing_pct}")
        sys.exit(1)

print("Data validation passed.")
sys.exit(0)
