# Real_World_Mlops — Demo Project

Overview
--------
This repository demonstrates a real-world MLOps workflow:
- Data and model artifact tracking with DVC
- Feature engineering with Feast (Feast.ipynb)
- Model training & experiments with DVC experiments
- Containerization (Dockerfile)
- Kubernetes deployment manifests (deployment.yaml, service.yaml, hpa.yml)
- CI/CD pipeline (GitHub Actions example provided)

Quickstart (local)
------------------
1. Clone:
   git clone https://github.com/mudgalma/Real_World_Mlops.git
   cd Real_World_Mlops

2. Create a virtual env & install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. DVC basics:
   dvc init                # if DVC not initialized
   # configure your remote (S3/MinIO/GCS)
   dvc remote add -d origin s3://my-bucket/path
   dvc pull                # pull data & models from remote
   dvc repro               # run pipeline stages
   dvc metrics show        # view metrics produced by pipeline

4. Run container locally:
   docker build -t mlops-demo:latest .
   docker run -p 8000:8000 mlops-demo:latest

5. Kubernetes (minikube / cluster):
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f hpa.yml

Feast feature store
-------------------
Open `Feast.ipynb` to see an example of defining and retrieving features.

Project structure
-----------------
- src/           # application & training code
- scripts/       # scripts used by DVC stages
- data/          # data root (raw, interim, processed)
- .dvc/          # DVC metadata
- deployment.yaml, service.yaml, hpa.yml  # K8s manifests
- Dockerfile
- requirements.txt

Improvements & Notes
--------------------
- Add `params.yaml` and `dvc.yaml` to formalize pipeline stages
- Use a DVC remote (S3/MinIO) and document credentials in CI secrets
- Add automated CI that runs dvc repro and unit tests
- Add a model registry (MLflow) or use DVC + tags
