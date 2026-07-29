# Real World MLOps

Real-world MLOps projects, tutorials and production-ready examples for deploying ML systems.

This repository collects practical MLOps projects, patterns, and end-to-end examples to help engineers, data scientists, and ML engineers build reliable, reproducible, and maintainable machine learning systems in production.

## Table of Contents
- [Overview](#overview)
- [Projects & Tutorials](#projects--tutorials)
- [Key Concepts Covered](#key-concepts-covered)
- [Getting Started](#getting-started)
- [Typical Project Structure](#typical-project-structure)
- [How to Use This Repository](#how-to-use-this-repository)
- [Running Examples / Quickstart](#running-examples--quickstart)
- [Recommended Tools & Tech Stack](#recommended-tools--tech-stack)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This repository is intended as a practical companion to learning and applying MLOps. Each folder contains an independent, end-to-end project or tutorial demonstrating a specific MLOps pattern — from local experimentation to automated training, CI/CD, deployment, monitoring, and operations.

Goals:
- Provide production-minded examples with clear, repeatable instructions.
- Demonstrate infrastructure-as-code, CI/CD, and observability for ML models.
- Show best practices for reproducibility, testing, and governance.

## Projects & Tutorials
Each top-level directory represents a project or tutorial. Typical examples include:
- Model training pipelines (batch and streaming)
- CI/CD pipelines for ML (testing, validation, model promotion)
- Containerized serving (Docker, FastAPI, TensorFlow Serving, TorchServe)
- Kubernetes deployments and autoscaling
- Feature store usage and data validation
- Model monitoring, drift detection, and alerting
- Experiment tracking and reproducible runs (MLflow, DVC)
- Serving, canary/blue-green deployments, and rollback strategies

(Explore directories for specific READMEs inside each project with step-by-step instructions.)

## Key Concepts Covered
- Reproducible training and deterministic experiments
- Data validation & schema checks (e.g., Great Expectations)
- Feature engineering pipelines and feature stores
- Model packaging & containerization
- CI for ML: unit tests, model validation tests, pipeline tests
- Deployment patterns: serverless, containers, k8s, inference clusters
- Observability: metrics, logs, tracing, and alerting for models
- Governance: model versioning, lineage, and promotion workflows

## Getting Started
Prerequisites (examples — each project may have specific requirements):
- Python 3.8+
- Docker
- git
- (Optional) Kubernetes (minikube / kind / a cloud k8s cluster)
- (Optional) MLflow, DVC, or other tools used by a specific project

Clone the repository:

```
git clone https://github.com/mudgalma/Real_World_Mlops.git
cd Real_World_Mlops
```

Then open the project folder you want to try and follow its README for per-project setup, dependencies, and run instructions.

## Typical Project Structure
A typical project folder follows a structure like:
- data/ — raw and sample datasets (or pointers to where to download them)
- src/ — training, evaluation, and preprocessing code
- notebooks/ — exploratory notebooks and demos
- infra/ — infrastructure-as-code (Terraform, Helm charts, k8s manifests)
- ci/ — CI pipeline configs and tests
- deployment/ — serving containers, manifests, and deployment steps
- tests/ — unit tests and model validation tests
- README.md — project-specific instructions and examples

## How to Use This Repository
1. Pick a project folder that matches what you'd like to learn (e.g., deployment, CI/CD, monitoring).
2. Read that folder's README for prerequisites and step-by-step instructions.
3. Run the included scripts or notebooks locally to understand the workflow.
4. Optionally, provision infrastructure in a sandbox environment (Docker/k8s) to test deployment scenarios.
5. Study CI config files to see how tests and model checks are automated.

## Running Examples / Quickstart
Each project includes a quickstart. A common pattern:
1. Create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Prepare data (scripts or download links inside the project)
4. Run training:

```
python src/train.py --config config/train.yaml
```

5. Start a local server for inference:

```
docker build -t project-serving .
docker run -p 8080:8080 project-serving
```

Refer to the specific project README for exact commands and environment variables.

## Recommended Tools & Tech Stack
- Experiment tracking: MLflow, Weights & Biases
- Data versioning: DVC
- CI/CD: GitHub Actions, Jenkins, or GitLab CI
- Containerization: Docker
- Orchestration: Kubernetes (Helm charts)
- Model serving: FastAPI, TorchServe, TensorFlow Serving, KFServing
- Monitoring: Prometheus, Grafana, Sentry (for errors), Evidently/WhyLogs (data/model monitoring)
- IaC: Terraform, Helm

## Contributing
Contributions are welcome. Ways to contribute:
- Add a new project or tutorial with clear steps and reproducible artifacts
- Improve existing README and examples
- Add tests, CI workflows, and deployment configurations
- Report issues with reproducibility or missing instructions
- Send a pull request following the repo contribution guidelines

Please open an issue first to discuss larger changes or new project ideas.

## License
Specify the license used by this repository (e.g., MIT). If no license exists, add one or clarify usage rights.

## Contact
Maintainer: mudgalma (GitHub)
For questions or suggestions, open an issue or create a pull request.

-----
Note: Some project directories may include additional, project-specific READMEs and detailed walkthroughs. Start there for hands-on instructions.
