import json, sys, os
import pandas as pd
from ydata_profiling import ProfileReport

CURRENT_METRICS = "reports/data_metrics.json"
BASELINE_METRICS = "reports/prev_data_metrics.json"
DRIFT_THRESHOLD = 0.2   # mean change > 20% = drift
CURRENT_DATA = "data/raw/dataset.csv"

# --- Load current data ---
df = pd.read_csv(CURRENT_DATA)

# --- Generate HTML drift report for humans ---
profile = ProfileReport(df, title="Data Drift Report", explorative=True)
profile.to_file("reports/drift_report.html")
print("✅ Saved human-friendly drift_report.html")

# --- JSON drift detection for automation ---
if not os.path.exists(CURRENT_METRICS):
    print("❌ No current metrics:", CURRENT_METRICS)
    sys.exit(2)

current = json.load(open(CURRENT_METRICS))

# First ever run → save baseline & exit cleanly
if not os.path.exists(BASELINE_METRICS):
    json.dump(current, open(BASELINE_METRICS, "w"), indent=2)
    print("✅ No baseline found — saved current metrics as baseline.")
    sys.exit(0)

baseline = json.load(open(BASELINE_METRICS))

# --- Check drift ---
for feature, cur_stats in current.get("summary", {}).items():
    base_stats = baseline.get("summary", {}).get(feature)
    
    if not base_stats:
        print(f"⚠️ New feature detected: {feature} → DRIFT")
        sys.exit(1)

    base_mean = base_stats.get("mean", 0)
    cur_mean = cur_stats.get("mean", 0)

    if base_mean == 0:
        continue

    rel_change = abs(cur_mean - base_mean) / abs(base_mean)

    if rel_change > DRIFT_THRESHOLD:
        print(f"🚨 DRIFT DETECTED for {feature}: change = {rel_change:.2f}")
        sys.exit(1)

print("✅ No significant drift detected.")
sys.exit(0)

