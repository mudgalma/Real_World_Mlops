import json, joblib, pandas as pd
from fastapi import FastAPI, Response, status
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
    __root__: dict

app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    import time
    time.sleep(2)
    app_state["is_ready"] = True

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

@app.post("/predict")
def predict(payload: Payload):
    data = payload.__root__
    df = pd.DataFrame([data])
    for c in FEATURES:
        if c not in df.columns:
            df[c] = None
    df = df[FEATURES]
    preds = model.predict(df)
    try:
        probs = getattr(model, "predict_proba", lambda X: None)(df)
        if probs is not None:
            probs = probs.tolist()
    except Exception:
        probs = None
    return {"prediction": preds.tolist(), "proba": probs}
