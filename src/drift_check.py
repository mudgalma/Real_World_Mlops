# scripts/drift_check.py
import json, sys, os, argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

parser = argparse.ArgumentParser()
parser.add_argument("--new", required=True, help="Path to new batch CSV")
parser.add_argument("--baseline", default="reports/data_metrics.json", help="Baseline metrics")
parser.add_argument("--min_new_rows", type=int, default=30, help="Minimum new rows to consider retraining")
parser.add_argument("--ks_p_threshold", type=float, default=0.01, help="KS-test p-value threshold")
args = parser.parse_args()

if not os.path.exists(args.new):
    print("Missing new batch:", args.new); sys.exit(2)

new_df = pd.read_csv(args.new)
if len(new_df) < args.min_new_rows:
    print(f"New batch too small ({len(new_df)} < {args.min_new_rows}) — store but do not trigger retrain.")
    sys.exit(0)  # not drift-trigger

# Load baseline (derived from full dataset metrics captured previously)
if not os.path.exists(args.baseline):
    print("Baseline metrics missing:", args.baseline)
    sys.exit(2)

base = json.load(open(args.baseline))
# load the actual full baseline dataset to compare distributions if needed (optional)
# For simplicity, compare numeric columns with KS test against baseline values from metrics file:
drift_detected = False
reasons = []

# compute lightweight stats by reloading baseline dataset if available
try:
    full_df = pd.read_csv("data/raw/dataset.csv")
except Exception:
    full_df = None

num_cols = new_df.select_dtypes(include=['number']).columns.tolist()
for c in num_cols:
    if full_df is not None and c in full_df.columns:
        stat = ks_2samp(full_df[c].dropna(), new_df[c].dropna())
        pval = stat.pvalue
        if pval < args.ks_p_threshold:
            drift_detected = True
            reasons.append(f"KS p<{args.ks_p_threshold} for {c} (p={pval:.3e})")

# Final decision
if drift_detected:
    print("DRIFT detected:", reasons)
    sys.exit(1)
else:
    print("No significant drift detected.")
    sys.exit(0)
