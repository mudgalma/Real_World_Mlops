# Real World MLOps — End-to-End MLOps Demo

Elevator pitch
--------------
A hands-on, production-oriented MLOps demo that covers the full lifecycle: reproducible data & model artifact management (DVC), feature engineering (Feast demo), experiments & model tracking (MLflow + training script), explainability, fairness and drift checks, containerized model serving (FastAPI + Docker), and Kubernetes deployment with autoscaling. Built to demonstrate practical MLOps skills you can walk a recruiter through in a short demo.

Highlights (what to show a recruiter)
-------------------------------------
- Reproducible pipelines and artifact management with DVC (project contains .dvc metadata).
- Production-ready training script (src/train.py) with:
  - MLflow tracking, autologging and model signature inference
  - Optional hyperparameter tuning (RandomizedSearchCV)
  - Clean preprocessing pipeline (ColumnTransformer + pipelines)
  - Stratified train/test split and robust logging
  - Optional label/data poisoning simulation for robustness testing
  - Optional OTEL tracing hooks (fail-safe)
- Model serving via FastAPI (api/app.py):
  - Health, readiness and liveness endpoints
  - Bootstraps model + schema from cloud storage (GCS)
  - Predict endpoint that returns predictions and probabilities
- Feature engineering demo with Feast (Feast.ipynb)
- Model quality checks & MLOps tooling under scripts/:
  - Drift detection with Evidently (scripts/evidently_drift.py)
  - Drift check utilities (scripts/drift_check.py)
  - Fairness checks (scripts/fairlearn_check.py)
  - Explainability helpers (scripts/explain.py)
  - Data validation & ingestion simulation (scripts/data_validation.py, scripts/simulate_ingest.py)
- Containerization & deployment:
  - Dockerfile to build container images
  - Kubernetes manifests: deployment.yaml, service.yaml, hpa.yml
- Designed to be demoed locally (docker/uvicorn) or in a cloud-native environment (GCS for artifacts, K8s for deployment).

Stack
-----
- Language(s): Python (training + serving), Jupyter (Feast demo), HTML for reports/notebooks
- Frameworks / runtimes: scikit-learn for modelling, FastAPI for serving, MLflow for tracking
- Notable libraries & tools: DVC, Feast, Evidently, Fairlearn, MLflow, OpenTelemetry (optional), Docker, Kubernetes

Repository structure (top-level)
-------------------------------

```
.dvc/                # DVC internal metadata
Dockerfile           # Container image
Feast.ipynb          # Notebook demo for Feast feature store
README.md            # (this file)
api/                 # FastAPI serving code + model artifacts location
  app.py
  model/             # model artifacts produced by training (pipeline.pkl, schema.json)
data/
  raw/               # raw dataset (DVC-managed)
deployment.yaml      # K8s Deployment manifest
hpa.yml              # HorizontalPodAutoscaler manifest
service.yaml         # K8s Service manifest
reports/             # (report outputs / notebooks / HTML)
requirements.txt
scripts/             # useful MLOps utilities: drift/fairness/explain/validation
src/                 # training & helper code
  train.py
  utils.py
  schema.py
```

How it fits together
--------------------
Workflow in short:
1. Data lives in data/raw (tracked by DVC).
2. Training executed by `src/train.py` produces a scikit-learn Pipeline and a schema file saved under `api/model/` and tracked/logged to MLflow.
3. The FastAPI app (api/app.py) loads pipeline and schema from cloud storage or local `api/model/` and serves /predict with health/readiness endpoints.
4. Monitoring & evaluation scripts (scripts/) enable drift detection, explainability, and fairness checks against production data.
5. Containerize with Dockerfile and deploy to Kubernetes using the provided manifests (with an HPA example).

Quickstart — demo locally (fast path)
------------------------------------
1. Clone and prepare environment
```bash
git clone https://github.com/mudgalma/Real_World_Mlops.git
cd Real_World_Mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional but recommended) Pull DVC-tracked data & models (configure remote first)
```bash
# if DVC remote is configured:
dvc pull
```

3. Train a model (example)
```bash
python src/train.py \
  --data_path data/raw/dataset.csv \
  --target target \
  --problem_type classification \
  --model_dir api/model \
  --test_size 0.2
```
What this produces:
- api/model/pipeline.pkl (saved scikit-learn Pipeline)
- api/model/schema.json (feature list + dtypes)
- MLflow run in local `mlruns/` with metrics: accuracy, f1_weighted (classification) or rmse, r2 (regression), plus train_time_sec

4. Run the API locally
```bash
# dev server
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080
# or with reload (dev)
uvicorn api.app:app --reload --host 0.0.0.0 --port 8080
```

5. Health & features
```bash
# Check health and get required feature names
curl http://localhost:8080/health
# Example output: {"status":"ok","features":["age","sex","..."]}
```

6. Predict (example)
- Build a JSON payload with the features returned above, e.g.:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":3,"trestbps":140, "chol":250, "target": 1}'
```
- The API returns predictions and probabilities (if supported by the model).

Docker & Kubernetes (deploy demo)
---------------------------------
- Build image:
```bash
docker build -t mlops-demo:latest .
```
- Run container:
```bash
docker run -p 8080:8080 mlops-demo:latest
```
- Deploy to Kubernetes (minikube or cluster):
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yml
```
Note: api/app.py expects model artifacts to be available in GCS (configurable). For local demos, ensure `api/model/pipeline.pkl` & `api/model/schema.json` exist inside the container or mock GCS downloads.

What to point out in an interview / talking points
-------------------------------------------------
- Reproducibility: DVC-managed data + artifacts and MLflow experiment tracking.
- Responsible ML: drift detection (Evidently), fairness checks (Fairlearn), and explainability scripts.
- Production readiness: structured JSON logging, liveness/readiness probes, autoscaling K8s manifests.
- Observability: optional OpenTelemetry hooks in training for distributed tracing.
- DevOps & cloud integration: GCS model bootstrap in the serving app, Docker + K8s manifests, and place for CI secrets for DVC remote.
- Code quality & engineering: single training script with clear CLI, modular preprocessing + model pipeline, and a serving contract (schema.json) used by the API.

Where to look (key files)
-------------------------
- Training & preprocessing: src/train.py
- Utilities: src/utils.py
- Serving: api/app.py
- Feature store demo: Feast.ipynb
- Monitoring & checks: scripts/evidently_drift.py, scripts/drift_check.py, scripts/fairlearn_check.py, scripts/explain.py
- Deployment: Dockerfile, deployment.yaml, service.yaml, hpa.yml
- Requirements: requirements.txt
- DVC metadata: .dvc/

Metrics & artifacts recorded
----------------------------
- MLflow metrics: accuracy, f1_weighted (classification) or rmse, r2 (regression), train_time_sec
- Artifacts: pipeline.pkl, schema.json, and MLflow model/artifact logs
- Logs: structured JSON logging to stdout for easy ingestion in log aggregators

Suggested additions to make this even more hireable
---------------------------------------------------
- Add `dvc.yaml` and `params.yaml` to make stages reproducible and automatable (README notes this).
- Add a small sample dataset (or script to download a public one) to make "one-click demos" trivial.
- CI: add GitHub Actions workflow that runs linting, unit tests, and `dvc repro` or quick smoke training.
- Add a short demo script or Makefile target that runs a full end-to-end local demo (train → serve → call predict).
- Add a short README demo GIF or recorded walkthrough for hiring managers.

Quick checklist for a recruiter demo (30–90s)
----------------------------------------------
1. git clone → pip install → dvc pull
2. Run: python src/train.py ...  (highlight MLflow metrics on console)
3. Run API: uvicorn api.app:app
4. curl /health then POST /predict
5. Open scripts/evidently_drift.py to show automated drift checks and explainability scripts

Skills demonstrated (keywords for resume)
-----------------------------------------
MLOps, DVC, MLflow, Feast, feature engineering, model deployment, FastAPI, Docker, Kubernetes, Horizontal Pod Autoscaler (HPA), model monitoring (Evidently), fairness testing (Fairlearn), explainability, Python, scikit-learn, structured logging, observability (OpenTelemetry), CI/CD-ready design.

Contact / Demo
--------------
If you'd like, I can:
- Produce a short Makefile to run an end-to-end local demo with one command.
- Add a sample dataset and `dvc.yaml` + `params.yaml` to make the pipeline reproducible out-of-the-box.
- Create a short demo script that runs training, serves the model, and posts a sample prediction.
