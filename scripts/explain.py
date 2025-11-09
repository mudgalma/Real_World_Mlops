#!/usr/bin/env python3
"""
Explainability Script (Production) - Corrected SHAP Implementation
Always downloads model + schema from GCS.

Usage:
  python scripts/explain.py --batch data/new_batch/new_batch.csv
  python scripts/explain.py  # Uses dataset.csv
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

def save_shap(model, X):
    """
    Compute SHAP exactly like iris notebook:
    1. Extract preprocessor and tree model from pipeline
    2. Transform X using preprocessor (SAME as training)
    3. Create TreeExplainer with transformed data
    4. Handle multiclass output (pick last class or use appropriate class)
    5. Plot with feature names from RAW X (before preprocessing)
    """
    
    # Extract components from pipeline
    preprocessor = model.named_steps["preprocess"]
    tree_model = model.named_steps["model"]
    
    # Transform X exactly as training (CRITICAL!)
    preprocessed_X = preprocessor.transform(X)
    
    # Create SHAP explainer with tree model
    explainer = shap.TreeExplainer(tree_model)
    
    # Get SHAP values for preprocessed data
    shap_output = explainer.shap_values(preprocessed_X)
    
    # ✅ NORMALIZE OUTPUT (handles both list and Explanation object)
    if isinstance(shap_output, list):
        shap_vals_list = shap_output
    else:
        # Explanation object - convert to list per class
        try:
            vals = np.asarray(shap_output.values)
            shap_vals_list = [vals[..., i] for i in range(vals.shape[-1])]
        except Exception:
            shap_vals_list = [shap_output]
    
    # ✅ FIX ARRAY DIMENSIONS (match X.shape[1])
    fixed_shap = []
    for arr in shap_vals_list:
        if arr.ndim == 1:
            arr = arr.reshape(-1, X.shape[1])
        if arr.shape[1] > X.shape[1]:
            arr = arr[:, :X.shape[1]]
        fixed_shap.append(arr)
    
    # ✅ PICK CLASS FOR BINARY (last class = disease positive)
    if len(fixed_shap) > 1:
        sv = fixed_shap[-1]  # Last class (disease = 1)
    else:
        sv = fixed_shap[0]
    
    # ✅ FEATURE NAMES = RAW feature names (before preprocessing)
    feature_names = list(X.columns)
    
    Path("reports").mkdir(exist_ok=True)
    
    # ✅ SUMMARY PLOT (exactly like iris notebook)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv,                      # SHAP values for class
        preprocessed_X,          # Preprocessed data (for coloring)
        feature_names=feature_names,  # Raw feature names
        show=False
    )
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=200)
    plt.close()
    
    print(f"✅ SHAP summary saved to reports/shap_summary.png (class: disease)")

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

    features = schema["columns"]
    
    # Generate SHAP (CORRECTED VERSION)
    save_shap(model, df[features])

if __name__ == "__main__":
    main()

