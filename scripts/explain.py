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


# def prepare_X(df, features):
#     """Create model-ready input."""
#     X = df[features].copy()

#     num_cols = X.select_dtypes(include=['int64', 'float64']).columns
#     cat_cols = X.select_dtypes(include=['object', 'category']).columns

#     if len(num_cols):
#         X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
#     if len(cat_cols):
#         X[cat_cols] = X[cat_cols].fillna("missing")
#         from sklearn.preprocessing import OrdinalEncoder
#         enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#         X[cat_cols] = enc.fit_transform(X[cat_cols])

#     return X


# def save_shap(model, X):
#     """Compute & save SHAP reports."""
#     tree_model = model.named_steps["model"]
#     pre_X = model.named_steps["preprocess"].transform(X)
#     expl = shap.TreeExplainer(tree_model)
#     shap_values = expl.shap_values(pre_X)

#     Path("reports").mkdir(exist_ok=True)

#     # Summary
#     plt.figure()
#     shap.summary_plot(shap_values, pre_X, show=False)
#     plt.savefig("reports/shap_summary.png", dpi=150)
#     plt.close()

#     # Per-feature
#     for col in X.columns:
#         try:
#             plt.figure()
#             shap.dependence_plot(col, shap_values, pre_X, show=False)
#             plt.savefig(f"reports/shap_feature_{col}.png", dpi=150)
#             plt.close()
#         except:
#             pass

#     print("✅ SHAP reports generated in /reports")
def save_shap(model, X):
    """Compute SHAP using the SAME preprocessor as training."""

    # Extract components from pipeline
    preprocessor = model.named_steps["preprocess"]
    tree_model = model.named_steps["model"]

    # Transform X exactly as training
    preprocessed_X = preprocessor.transform(X)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(preprocessed_X)

    # If multiclass, pick last class
    if isinstance(shap_values, list):
        sv = shap_values[-1]
    else:
        sv = shap_values

    # Features = raw feature names AFTER removing target/sno
    feature_names = list(X.columns)

    Path("reports").mkdir(exist_ok=True)

    # ✅ Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, preprocessed_X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150)
    plt.close()

    # ✅ Feature importance CSV
    # mean_abs = np.mean(np.abs(sv), axis=0)
    # imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    # imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
    # imp_df.to_csv("reports/shap_feature_importance.csv", index=False)

    # ✅ Per-feature plots
    # for feat in feature_names:
    #     try:
    #         plt.figure(figsize=(6, 3))
    #         shap.dependence_plot(feat,
    #                               sv,
    #                               preprocessed_X,
    #                               feature_names=feature_names,
    #                               show=False)
    #         plt.tight_layout()
    #         plt.savefig(f"reports/shap_feature_{feat}.png", dpi=120)
    #         plt.close()
    #     except Exception:
    #         pass

    # print("✅ SHAP reports generated in /reports")



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
    features = all_cols 

   

    save_shap(model, df[features])


if __name__ == "__main__":
    main()
