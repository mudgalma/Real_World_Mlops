# #!/usr/bin/env python3
# """
# Fairness Report Script (Production)
# Always downloads model + schema from GCS.
# """

# import os
# import json
# import tempfile
# from pathlib import Path

# import pandas as pd
# import joblib
# from fairlearn.metrics import MetricFrame, selection_rate
# from google.cloud import storage

# BUCKET = "heart-disease-mlops-data"
# MODEL_PATH = "models/pipeline.pkl"
# SCHEMA_PATH = "models/schema.json"


# def download_from_gcs(blob_path: str, dst: str):
#     client = storage.Client()
#     bucket = client.bucket(BUCKET)
#     blob = bucket.blob(blob_path)
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     blob.download_to_filename(dst)
#     return dst


# def load_model_schema():
#     tmp = Path(tempfile.mkdtemp(prefix="fair_gcs_"))
#     model_file = tmp / "pipeline.pkl"
#     schema_file = tmp / "schema.json"

#     print("⬇️ Downloading model and schema from GCS…")
#     download_from_gcs(MODEL_PATH, str(model_file))
#     download_from_gcs(SCHEMA_PATH, str(schema_file))

#     model = joblib.load(model_file)
#     schema = json.load(open(schema_file))

#     return model, schema


# def main():
#     model, schema = load_model_schema()

#     # Load dataset (same used in training)
#     df = pd.read_csv("data/raw/dataset.csv")

#     # ✅ Schema now contains ONLY features
#     features = schema["columns"]

#     # ✅ Target is ONLY in CSV, not in schema
#     if "target" not in df.columns:
#         raise ValueError("Dataset does not contain target column")

#     target = df["target"]
#     X = df[features]
#     y_pred = model.predict(X)

#     # ✅ Sensitive attribute (you said AGE is sensitive)
#     sensitive = df["age"]

#     # Compute fairness metrics
#     mf = MetricFrame(
#         metrics=selection_rate,
#         y_true=target,
#         y_pred=y_pred,
#         sensitive_features=sensitive
#     )

#     Path("reports").mkdir(exist_ok=True)
#     mf.by_group.to_csv("reports/fairness_metrics.csv")

#     print("✅ Fairness metrics saved → reports/fairness_metrics.csv")

#!/usr/bin/env python3
"""
Fairness Report Script (Production)
Always downloads model + schema from GCS.
Now handles continuous AGE by binning into discrete groups.
"""

import os
import json
import tempfile
from pathlib import Path

import pandas as pd
import joblib
from fairlearn.metrics import MetricFrame, selection_rate
from google.cloud import storage

BUCKET = "heart-disease-mlops-data"
MODEL_PATH = "models/pipeline.pkl"
SCHEMA_PATH = "models/schema.json"


def download_from_gcs(blob_path: str, dst: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    blob.download_to_filename(dst)
    return dst


def load_model_schema():
    tmp = Path(tempfile.mkdtemp(prefix="fair_gcs_"))
    model_file = tmp / "pipeline.pkl"
    schema_file = tmp / "schema.json"

    print("⬇️ Downloading model and schema from GCS…")
    download_from_gcs(MODEL_PATH, str(model_file))
    download_from_gcs(SCHEMA_PATH, str(schema_file))

    model = joblib.load(model_file)
    schema = json.load(open(schema_file))

    return model, schema


def main():
    model, schema = load_model_schema()

    # Load dataset (same used in training)
    df = pd.read_csv("data/raw/dataset.csv")

    # ✅ Schema now contains ONLY features
    features = schema["columns"]

    # ✅ Target is ONLY in CSV, not in schema
    if "target" not in df.columns:
        raise ValueError("Dataset does not contain target column")

    target = df["target"]
    X = df[features]
    y_pred = model.predict(X)

    # ✅ Handle continuous AGE → convert to binned categories
    if "age" not in df.columns:
        raise ValueError("Sensitive feature 'age' not found in dataset")

    # Create 4 age bins (customize as needed)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 120],
        labels=["<30", "30-45", "45-60", "60+"],
        include_lowest=True
    )

    print("🧩 Created age bins:", df["age_group"].unique())

    # Compute fairness metrics
    mf = MetricFrame(
        metrics=selection_rate,
        y_true=target,
        y_pred=y_pred,
        sensitive_features=df["age_group"]
    )

    Path("reports").mkdir(exist_ok=True)
    mf.by_group.to_csv("reports/fairness_metrics.csv")

    print("✅ Fairness metrics saved → reports/fairness_metrics.csv")


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     main()
