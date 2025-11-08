import json
import joblib
import pandas as pd
import logging
import os
import time
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from google.cloud import storage

# ----------------------------
# Configuration
# ----------------------------
BUCKET_NAME = "heart-disease-mlops-data"
MODEL_BLOB = "models/pipeline.pkl"
SCHEMA_BLOB = "models/schema.json"
LOCAL_MODEL_PATH = "api/model/model.pkl"
LOCAL_SCHEMA_PATH = "api/model/schema.json"

app = FastAPI(title="Heart Disease ML API")

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-api")

# ----------------------------
# Download model & schema from GCS
# ----------------------------
def download_from_gcs(bucket_name, source_blob, destination_file):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(destination_file)
        logger.info(f"Downloaded {source_blob} to {destination_file}")
    except Exception as e:
        logger.error(f"Error downloading {source_blob}: {e}")
        raise

def load_model_and_schema():
    download_from_gcs(BUCKET_NAME, MODEL_BLOB, LOCAL_MODEL_PATH)
    download_from_gcs(BUCKET_NAME, SCHEMA_BLOB, LOCAL_SCHEMA_PATH)

    model = joblib.load(LOCAL_MODEL_PATH)
    with open(LOCAL_SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return model, schema

# ----------------------------
# App state & startup
# ----------------------------
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting up app and loading model from GCS...")
        global model, FEATURES
        model, schema = load_model_and_schema()
        FEATURES = schema.get("columns", schema.get("feature_names", []))
        app_state["is_ready"] = True
        logger.info("Model successfully loaded and app ready.")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        app_state["is_ready"] = False

# ----------------------------
# Health endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES}

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

# ----------------------------
# Prediction endpoint
# ----------------------------
class Payload(BaseModel):
    __root__: dict

@app.post("/predict")
def predict(payload: Payload):
    if not app_state["is_ready"]:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    data = payload.__root__
    df = pd.DataFrame([data])
    for c in FEATURES:
        if c not in df.columns:
            df[c] = None
    df = df[FEATURES]

    try:
        preds = model.predict(df)
        probs = getattr(model, "predict_proba", lambda X: None)(df)
        logger.info(f"Prediction: {preds.tolist()} | Input: {data}")
        return {"prediction": preds.tolist(), "proba": probs.tolist() if probs is not None else None}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8081, reload=True)
