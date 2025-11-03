import json, sys, os, argparse
import pandas as pd
from ydata_profiling import ProfileReport


CURRENT_METRICS = "reports/data_metrics.json"
BASELINE_METRICS = "reports/prev_data_metrics.json"
DRIFT_THRESHOLD = 0  # 20% change triggers drift


def load_stats(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def compare_stats(current, baseline):
    """Return True if drift detected."""
    for feature, cur_stats in current.get("summary", {}).items():
        base_stats = baseline.get("summary", {}).get(feature)
        
        if not base_stats:
            print(f"⚠️ New feature detected: {feature} → DRIFT")
            return True

        base_mean = base_stats.get("mean", 0)
        cur_mean = cur_stats.get("mean", 0)

        if base_mean == 0:
            continue

        rel_change = abs(cur_mean - base_mean) / abs(base_mean)

        if rel_change > DRIFT_THRESHOLD:
            print(f"🚨 DRIFT DETECTED for {feature}: change = {rel_change:.2f}")
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "ingest"], required=True)
    parser.add_argument("--batch", help="Path to new batch for ingestion mode")
    args = parser.parse_args()

    # --- Determine dataset path ---
    if args.mode == "ci":
        data_path = "data/raw/dataset.csv"
    else:
        if not args.batch:
            print("❌ Ingest mode requires --batch file")
            sys.exit(2)
        data_path = args.batch

    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        sys.exit(2)

    # --- Compute stats + save HTML report ---
    df = pd.read_csv(data_path)
    print(f"✅ Loaded data: {data_path}")

    profile = ProfileReport(df, title="Data Drift Report", explorative=True)
    profile.to_file("reports/drift_report.html")
    print("✅ Saved drift_report.html")

    # --- Metrics must already exist (from validation step) ---
    if not os.path.exists(CURRENT_METRICS):
        print(f"❌ Missing metrics file: {CURRENT_METRICS}")
        sys.exit(2)

    current_stats = load_stats(CURRENT_METRICS)
    baseline_stats = load_stats(BASELINE_METRICS)

    # First run → create baseline
    if baseline_stats is None:
        json.dump(current_stats, open(BASELINE_METRICS, "w"), indent=2)
        print("✅ No baseline found. Saved current stats as baseline.")
        sys.exit(0)

    # Compare
    drift = compare_stats(current_stats, baseline_stats)

    if drift:
        print("🚨 Drift detected.")
        sys.exit(1)

    print("✅ No drift detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
