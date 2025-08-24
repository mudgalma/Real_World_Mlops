# api/app.py
import json, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="OPPE Model API")

# Load model & schema
MODEL_PATH = "model/model.pkl"
SCHEMA_PATH = "model/schema.json"
model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)
FEATURES = schema["feature_names"]

class Payload(BaseModel):
    # Accepts arbitrary feature dict; we'll filter by schema
    __root__: dict

@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES}

@app.post("/predict")
def predict(payload: Payload):
    data = payload.__root__
    df = pd.DataFrame([data])
    # ensure all expected features exist; missing → fill NA
    for c in FEATURES:
        if c not in df.columns: df[c] = None
    df = df[FEATURES]
    preds = model.predict(df)
    try:
        probs = getattr(model, "predict_proba", lambda X: None)(df)
        if probs is not None:
            probs = probs.tolist()
    except Exception:
        probs = None
    return {"prediction": preds.tolist(), "proba": probs}
