# scripts/evidently_drift.py
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path
import sys

BASE = "data/raw/dataset.csv"
NEW = "data/new_batch/new_batch.csv"
OUT = "reports/drift_report.html"

Path("reports").mkdir(exist_ok=True)

if not Path(BASE).exists():
    print("Missing base dataset:", BASE); sys.exit(2)

if not Path(NEW).exists():
    print("Missing new batch:", NEW); sys.exit(2)

df_base = pd.read_csv(BASE)
df_new = pd.read_csv(NEW)

# Full Evidently Drift Report
report = Report(metrics=[DataDriftPreset()])
result  = report.run(reference_data=df_new, current_data=df_new)
result.save_html("reports/drift_report.html")
print("✅ Drift report saved: reports/drift_report.html")
print(f"Evidently drift report saved to {OUT}")
