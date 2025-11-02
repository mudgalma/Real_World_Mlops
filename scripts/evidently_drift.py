# # scripts/evidently_drift.py
# import argparse
# import json
# from pathlib import Path
# import pandas as pd
# from evidently import Report 
# from evidently.presets import DataDriftPreset
# import sys

# # -----------------------------------------
# # Argument parser
# # -----------------------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--mode",
#     choices=["ci", "ingest"],
#     required=True,
#     help="Mode: 'ci' for full dataset drift, 'ingest' for new batch drift"
# )
# parser.add_argument(
#     "--batch",
#     type=str,
#     help="Path to new batch CSV (required for ingest mode)"
# )

# args = parser.parse_args()

# # -----------------------------------------
# # Paths
# # -----------------------------------------
# BASE_DATA = Path("data/raw/dataset.csv")
# REPORT_DIR = Path("reports")
# REPORT_DIR.mkdir(exist_ok=True)

# HTML_OUT = REPORT_DIR / "evidently_drift_report.html"
# JSON_OUT = REPORT_DIR / "evidently_drift.json"

# # -----------------------------------------
# # Validate base dataset
# # -----------------------------------------
# if not BASE_DATA.exists():
#     print(f"❌ ERROR: Base dataset missing at {BASE_DATA}")
#     sys.exit(2)

# df_base = pd.read_csv(BASE_DATA)

# # ==========================================================
# # ✅ MODE 1: CI MODE (compare current dataset to last baseline)
# # ==========================================================
# if args.mode == "ci":
#     print("🔵 Running Evidently Drift in CI mode...")

#     # load current dataset (= base dataset)
#     df_cur = df_base.copy()

#     # Prepare Evidently drift report
#     report = Report(metrics=[DataDriftPreset()])
#     result = report.run(reference_data=df_base, current_data=df_cur)
#     result.save_html(str(HTML_OUT))
#     # report = Report(metrics=[DataDriftPreset()])
#     # report.run(reference_data=df_base, current_data=df_cur)
#     # report.save_html(str(HTML_OUT))

#     # Save JSON summary (contains drift metrics)
#     json_result = report.as_dict()
#     with open(JSON_OUT, "w") as f:
#         json.dump(json_result, f, indent=2)

#     print(f"✅ CI drift report saved → {HTML_OUT}")
#     print(f"✅ CI drift summary saved → {JSON_OUT}")
#     sys.exit(0)

# # ==========================================================
# # ✅ MODE 2: INGEST MODE (compare base dataset with new batch)
# # ==========================================================
# elif args.mode == "ingest":
#     if not args.batch:
#         print("❌ ERROR: --batch must be provided in ingest mode.")
#         sys.exit(2)

#     new_batch_path = Path(args.batch)
#     if not new_batch_path.exists():
#         print(f"❌ ERROR: New batch file missing: {new_batch_path}")
#         sys.exit(2)

#     print("🟢 Running Evidently Drift in INGEST mode...")
#     df_new = pd.read_csv(new_batch_path)

#     # Prepare Evidently report
#     report = Report(metrics=[DataDriftPreset()])
#     result = report.run(reference_data=df_base, current_data=df_cur)
#     result.save_html(str(HTML_OUT))

#     # Save JSON summary
#     json_result = report.as_dict()
#     with open(JSON_OUT, "w") as f:
#         json.dump(json_result, f, indent=2)

#     print(f"✅ Ingest drift report saved → {HTML_OUT}")
#     print(f"✅ Ingest drift summary saved → {JSON_OUT}")

#     # ✅ Output a clear drift status for GitHub Actions
#     # If drift score > 0.1 → exit 1 so ingest.yml can trigger retrain
#     drift_score = json_result["metrics"][0]["result"].get("data_drift", 0)
#     print(f"🔎 Drift score = {drift_score}")

#     if drift_score > 0.1:
#         print("🚨 DRIFT DETECTED (score > 0.10)")
#         sys.exit(1)

#     print("✅ No significant drift.")
#     sys.exit(0)
#!/usr/bin/env python3
# scripts/evidently_drift.py
import argparse
import json
from pathlib import Path
import pandas as pd
import sys
from evidently import Report 
from evidently.presets import DataDriftPreset


def save_html_and_json(result_obj, html_path: Path, json_path: Path):
    """Save HTML and JSON from an Evidently run result or Report (defensive)."""
    # Save HTML
    if hasattr(result_obj, "save_html"):
        result_obj.save_html(str(html_path))
    else:
        # some versions put save_html on Report, so try that (defensive)
        try:
            Report.save_html(result_obj, str(html_path))  # type: ignore
        except Exception:
            print("⚠️ Could not save HTML via result_obj.save_html or Report.save_html")

    # Get JSON dict
    json_result = {}
    if hasattr(result_obj, "as_dict"):
        json_result = result_obj.as_dict()
    else:
        # maybe result_obj is Report - try Report.as_dict
        try:
            json_result = Report.as_dict(result_obj)  # type: ignore
        except Exception:
            json_result = {}

    # Write JSON to file
    try:
        with open(json_path, "w") as fh:
            json.dump(json_result, fh, indent=2)
    except Exception as e:
        print("⚠️ Failed to write JSON:", e)

    return json_result


def extract_drift_score(json_result: dict) -> float:
    """
    Best-effort extraction of numeric drift score (0..1) from Evidently JSON.
    Evidently JSON formats vary across versions; we check likely places.
    """
    try:
        metrics = json_result.get("metrics", []) or []
        for m in metrics:
            r = m.get("result", {}) or {}
            # common keys
            # 1) direct drift_score
            if isinstance(r.get("drift_score"), (int, float)):
                return float(r["drift_score"])
            # 2) data_drift as dict with 'drift_share' or similar
            dd = r.get("data_drift")
            if isinstance(dd, dict):
                for k in ("drift_share", "drift_score", "drift_percentage"):
                    if k in dd and isinstance(dd[k], (int, float)):
                        return float(dd[k])
            # 3) some versions store numeric values under nested stats
            # attempt to find first float in r
            for v in r.values():
                if isinstance(v, (int, float)):
                    return float(v)
    except Exception:
        pass
    # fallback: if data_drift boolean present, return 1.0 for True else 0.0
    try:
        for m in json_result.get("metrics", []):
            if isinstance(m.get("result", {}).get("data_drift"), bool):
                return 1.0 if m["result"]["data_drift"] else 0.0
    except Exception:
        pass

    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "ingest"], required=True)
    parser.add_argument("--batch", type=str, help="Path to new batch CSV (required for ingest mode)")
    parser.add_argument("--drift-threshold", type=float, default=0.10,
                        help="Threshold (0..1) above which ingest mode signals drift (default 0.10)")
    args = parser.parse_args()

    BASE_DATA = Path("data/raw/dataset.csv")
    REPORT_DIR = Path("reports")
    REPORT_DIR.mkdir(exist_ok=True)
    HTML_OUT = REPORT_DIR / "evidently_drift_report.html"
    JSON_OUT = REPORT_DIR / "evidently_drift.json"

    if not BASE_DATA.exists():
        print(f"❌ ERROR: Base dataset missing at {BASE_DATA}")
        sys.exit(2)

    df_base = pd.read_csv(BASE_DATA)

    # CI mode: run report on full dataset (baseline vs itself)
    if args.mode == "ci":
        print("🔵 Running Evidently (CI mode) — baseline generation / sanity check")
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=df_base, current_data=df_base)

        json_result = save_html_and_json(result, HTML_OUT, JSON_OUT)
        print(f"✅ CI HTML: {HTML_OUT}")
        print(f"✅ CI JSON: {JSON_OUT}")
        sys.exit(0)

    # INGEST mode: run report comparing base -> new batch
    if args.mode == "ingest":
        if not args.batch:
            print("❌ ERROR: --batch is required in ingest mode")
            sys.exit(2)
        new_batch_path = Path(args.batch)
        if not new_batch_path.exists():
            print(f"❌ ERROR: New batch file missing: {new_batch_path}")
            sys.exit(2)

        print("🟢 Running Evidently (INGEST mode) — comparing base -> new batch")
        df_new = pd.read_csv(new_batch_path)

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=df_base, current_data=df_new)

        json_result = save_html_and_json(result, HTML_OUT, JSON_OUT)
        print(f"✅ Ingest HTML: {HTML_OUT}")
        print(f"✅ Ingest JSON: {JSON_OUT}")

        drift_score = extract_drift_score(json_result)
        print(f"🔎 Derived drift score = {drift_score:.4f} (threshold = {args.drift_threshold:.4f})")

        if drift_score > args.drift_threshold:
            print("🚨 DRIFT detected (score > threshold)")
            sys.exit(1)

        print("✅ No significant drift detected.")
        sys.exit(0)

    print("❌ Unknown mode")
    sys.exit(2)


if __name__ == "__main__":
    main()

