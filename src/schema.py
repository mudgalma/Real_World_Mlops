# src/schema.py
import json
import pandas as pd

def infer_feature_schema(X: pd.DataFrame):
    schema = {
        "feature_names": list(X.columns),
        "dtypes": {c: str(X[c].dtype) for c in X.columns}
    }
    return schema

def save_feature_schema(schema, path: str):
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
