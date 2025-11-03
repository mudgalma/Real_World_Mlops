#!/usr/bin/env python3
"""
Explainability Script (Production)
Always downloads model + schema from GCS.

Usage:
  python scripts/explain.py --batch data/new_batch/new_batch.csv   # ingest pipeline
  python scripts/explain.py                                        # CI pipeline
"""

import os
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from google.cloud import storage


BUCKET = "heart-disease-mlops-data"
MODEL_PATH = "models/pipeline.pkl"
SCHEMA_PATH = "models/schema.json"


def download_from_gcs(blob_path: str, dst: str):
    """Download file from GCS."""
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    blob.download_to_filename(dst)
    return dst


def load_model_schema():
    """Always load model + schema from GCS."""
    tmp = Path(tempfile.mkdtemp(prefix="explain_gcs_"))
    model_local = tmp / "pipeline.pkl"
    schema_local = tmp / "schema.json"

    print("⬇️  Downloading model & schema from GCS...")
    download_from_gcs(MODEL_PATH, str(model_local))
    download_from_gcs(SCHEMA_PATH, str(schema_local))

    model = joblib.load(model_local)
    with open(schema_local) as f:
        schema = json.load(f)

    return model, schema


def prepare_X(df, features):
    """Create model-ready input."""
    X = df[features].copy()

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(num_cols):
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    if len(cat_cols):
        X[cat_cols] = X[cat_cols].fillna("missing")
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    return X


def save_shap(model, X):
    """Compute & save SHAP reports."""
    Path("reports").mkdir(exist_ok=True)

    print("⚡ Computing SHAP...")
    expl = shap.TreeExplainer(model)
    shap_values = expl.shap_values(X)

    # Summary
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("reports/shap_summary.png", dpi=150)
    plt.close()

    # Per-feature
    for col in X.columns:
        try:
            plt.figure()
            shap.dependence_plot(col, shap_values, X, show=False)
            plt.savefig(f"reports/shap_feature_{col}.png", dpi=150)
            plt.close()
        except:
            pass

    print("✅ SHAP reports generated in /reports")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default=None)
    args = parser.parse_args()

    # Load model + schema (ALWAYS FROM GCS NOW)
    model, schema = load_model_schema()

    # Load data
    if args.batch and Path(args.batch).exists():
        df = pd.read_csv(args.batch)
        print("📦 Using new batch for explainability")
    else:
        df = pd.read_csv("data/raw/dataset.csv")
        print("📦 Using dataset.csv for explainability")

    all_cols = schema["columns"]
    if "target" in all_cols:
        features = [c for c in all_cols if c != "target"]
    else:
        raise ValueError("schema.json is missing 'target' in columns list")

    X = prepare_X(df, features)
    X = prepare_X(df, features)

    save_shap(model, X)


if __name__ == "__main__":
    main()
