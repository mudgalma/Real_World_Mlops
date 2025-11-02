# scripts/evidently_drift.py
import argparse
import json
from pathlib import Path
import pandas as pd
from evidently import Report 
from evidently.presets import DataDriftPreset
import sys

# -----------------------------------------
# Argument parser
# -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["ci", "ingest"],
    required=True,
    help="Mode: 'ci' for full dataset drift, 'ingest' for new batch drift"
)
parser.add_argument(
    "--batch",
    type=str,
    help="Path to new batch CSV (required for ingest mode)"
)

args = parser.parse_args()

# -----------------------------------------
# Paths
# -----------------------------------------
BASE_DATA = Path("data/raw/dataset.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

HTML_OUT = REPORT_DIR / "evidently_drift_report.html"
JSON_OUT = REPORT_DIR / "evidently_drift.json"

# -----------------------------------------
# Validate base dataset
# -----------------------------------------
if not BASE_DATA.exists():
    print(f"❌ ERROR: Base dataset missing at {BASE_DATA}")
    sys.exit(2)

df_base = pd.read_csv(BASE_DATA)

# ==========================================================
# ✅ MODE 1: CI MODE (compare current dataset to last baseline)
# ==========================================================
if args.mode == "ci":
    print("🔵 Running Evidently Drift in CI mode...")

    # load current dataset (= base dataset)
    df_cur = df_base.copy()

    # Prepare Evidently drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_base, current_data=df_cur)
    report.save_html(str(HTML_OUT))

    # Save JSON summary (contains drift metrics)
    json_result = report.as_dict()
    with open(JSON_OUT, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"✅ CI drift report saved → {HTML_OUT}")
    print(f"✅ CI drift summary saved → {JSON_OUT}")
    sys.exit(0)

# ==========================================================
# ✅ MODE 2: INGEST MODE (compare base dataset with new batch)
# ==========================================================
elif args.mode == "ingest":
    if not args.batch:
        print("❌ ERROR: --batch must be provided in ingest mode.")
        sys.exit(2)

    new_batch_path = Path(args.batch)
    if not new_batch_path.exists():
        print(f"❌ ERROR: New batch file missing: {new_batch_path}")
        sys.exit(2)

    print("🟢 Running Evidently Drift in INGEST mode...")
    df_new = pd.read_csv(new_batch_path)

    # Prepare Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_base, current_data=df_new)
    report.save_html(str(HTML_OUT))

    # Save JSON summary
    json_result = report.as_dict()
    with open(JSON_OUT, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"✅ Ingest drift report saved → {HTML_OUT}")
    print(f"✅ Ingest drift summary saved → {JSON_OUT}")

    # ✅ Output a clear drift status for GitHub Actions
    # If drift score > 0.1 → exit 1 so ingest.yml can trigger retrain
    drift_score = json_result["metrics"][0]["result"].get("data_drift", 0)
    print(f"🔎 Drift score = {drift_score}")

    if drift_score > 0.1:
        print("🚨 DRIFT DETECTED (score > 0.10)")
        sys.exit(1)

    print("✅ No significant drift.")
    sys.exit(0)

