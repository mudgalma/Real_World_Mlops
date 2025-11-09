# src/train.py
"""
Production-ready training script.

Usage (example):
  python src/train.py --target target --problem_type classification --model_dir api/model --test_size 0.2
"""
import os
import json
import time
import argparse
import warnings
from typing import Optional
import logging
import sys
import json
from contextlib import nullcontext
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# OTEL tracing (safe / optional)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    OTEL_AVAILABLE = True
except Exception:
    OTEL_AVAILABLE = False

# local imports
from src.utils import apply_poisoning  # allowed but not used by default
from src.schema import infer_feature_schema # keep if present

warnings.filterwarnings("ignore")
# Structured logging setup
logger = logging.getLogger("mlops-train")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

def setup_tracer(project_id: Optional[str] = None):
    """Set up OTEL tracing if available and env allows it. Fail-safe (no exception)."""
    if not OTEL_AVAILABLE:
        return None
    try:
        provider = TracerProvider()
        exporter = CloudTraceSpanExporter(project_id=project_id) if project_id else None
        if exporter:
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            return trace.get_tracer(__name__)
    except Exception:
        # don't crash training if OTEL misconfigured
        return None
    return None


def build_preprocessor(X: pd.DataFrame):
    """Infer numeric and categorical cols and return ColumnTransformer."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return preprocessor, numeric_cols, cat_cols


def build_model(problem_type: str, random_state: int = 42):
    if problem_type == "classification":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    else:
        return RandomForestRegressor(random_state=random_state, n_jobs=-1)


def save_pipeline(pipeline: Pipeline, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "pipeline.pkl")
    joblib.dump(pipeline, out_path)
    return out_path


def save_schema_from_df(df: pd.DataFrame, out_path: str):
    target_col = "target"
    schema = {
        "columns": [c for c in df.columns if c != target_col],
        "dtypes": {c: str(df[c].dtype) for c in df.columns if c != target_col}
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
    return out_path


def train_main(
    data_path: str,
    target: str,
    problem_type: str,
    model_dir: str = "api/model",
    test_size: float = 0.2,
    random_state: int = 42,
    poison: Optional[str] = None,
    tune: bool = False,
):
    tracer = setup_tracer()  # safe

    try:
        # Data loading
        with tracer.start_as_current_span("data_loading") if tracer else nullcontext():
            logger.info(json.dumps({"event": "data_loading", "data_path": data_path}))
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}. Run `dvc pull` first.")
            df = pd.read_csv(data_path)
            print(df['target'].value_counts())
            logger.info(json.dumps({"event": "data_loaded", "rows": len(df)}))

        # MLflow setup
                # MLflow setup
        EXPERIMENT_NAME = "real_world_heart_mlop"
        TRACKING_URI = "mlruns"  # or "mlruns" for local

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)


        # Target check
        if target not in df.columns:
            logger.error(json.dumps({"event": "target_missing", "target": target, "columns": df.columns.tolist()}))
            raise ValueError(f"Target '{target}' not found in data columns: {df.columns.tolist()}")

        # Optional poisoning
        if poison:
            with tracer.start_as_current_span("data_poisoning") if tracer else nullcontext():
                logger.info(json.dumps({"event": "data_poisoning", "spec": poison}))
                df = apply_poisoning(df, target, problem_type, poison)

        # Drop rows with missing target
        df = df.dropna(subset=[target]).reset_index(drop=True)

        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Classification label encoding
        if problem_type == "classification" and (y.dtype == "object" or str(y.dtype).startswith("category")):
            if set(y.unique()) <= {"yes", "no"}:
                y = y.map({"no": 0, "yes": 1})
            else:
                y, _ = pd.factorize(y)

        # Save schema for API
        save_schema_from_df(X, os.path.join(model_dir, "schema.json"))

        # Split
        stratify = None if problem_type == "regression" else y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Build preprocessor
        preprocessor, numeric_cols, cat_cols = build_preprocessor(X_train)

        # Assemble pipeline
        model = build_model(problem_type=problem_type, random_state=random_state)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        # MLflow autolog
        mlflow.sklearn.autolog(log_input_examples=True, disable=False, silent=True)

        with mlflow.start_run(run_name=f"{problem_type}_rf_run"):
            mlflow.log_params({
                "problem_type": problem_type,
                "test_size": test_size,
                "random_state": random_state,
                "poison": poison or "none",
                "tune": bool(tune),
            })

            # Hyperparameter tuning
            if tune:
                with tracer.start_as_current_span("hyperparameter_tuning") if tracer else nullcontext():
                    logger.info(json.dumps({"event": "hyperparameter_tuning_start"}))
                    param_dist = {
                        "model__n_estimators": [10, 30, 50],
                        "model__max_depth": [None, 5, 10],
                        "model__min_samples_split": [2, 5],
                    }
                    search = RandomizedSearchCV(
                        pipeline,
                        param_distributions=param_dist,
                        n_iter=3,
                        cv=3,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=0,
                    )
                    t0 = time.time()
                    search.fit(X_train, y_train)
                    train_time = time.time() - t0
                    best = search.best_estimator_
                    pipeline = best
                    mlflow.log_metric("train_time_sec", train_time)
                    mlflow.log_params({"tuned": True, **search.best_params_})
                    logger.info(json.dumps({"event": "hyperparameter_tuning_end", "best_params": search.best_params_, "train_time_sec": train_time}))
            else:
                with tracer.start_as_current_span("model_training") if tracer else nullcontext():
                    logger.info(json.dumps({"event": "model_training_start"}))
                    t0 = time.time()
                    pipeline.fit(X_train, y_train)
                    train_time = time.time() - t0
                    mlflow.log_metric("train_time_sec", train_time)
                    logger.info(json.dumps({"event": "model_training_end", "train_time_sec": train_time}))

            # Predictions and metrics
            with tracer.start_as_current_span("model_evaluation") if tracer else nullcontext():
                preds = pipeline.predict(X_test)
                if problem_type == "classification":
                    logger.info(f"y_test sample: {y_test[:10].tolist()}")
                    logger.info(f"preds sample: {preds[:10].tolist()}")
  
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
                    logger.info(json.dumps({"event": "model_evaluation", "accuracy": acc, "f1_weighted": f1}))
                    print(f"Accuracy={acc:.4f}  F1={f1:.4f}")
                else:
                    rmse = mean_squared_error(y_test, preds, squared=False)
                    r2 = r2_score(y_test, preds)
                    mlflow.log_metrics({"rmse": rmse, "r2": r2})
                    logger.info(json.dumps({"event": "model_evaluation", "rmse": rmse, "r2": r2}))
                    print(f"RMSE={rmse:.4f}  R2={r2:.4f}")

            # Save pipeline locally and log model with signature
            os.makedirs(model_dir, exist_ok=True)
            pipeline_path = save_pipeline(pipeline, model_dir)


            # Infer signature using a small sample from training input
            try:
                signature = infer_signature(X_train.iloc[:5], pipeline.predict(X_train.iloc[:5]))
            except Exception:
                signature = None

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=None,
                signature=signature,
            )

            # Also save artifacts locally for API usage
            mlflow.log_artifact(pipeline_path, artifact_path="artifacts")
            schema_path = os.path.join(model_dir, "schema.json")
            mlflow.log_artifact(schema_path, artifact_path="artifacts")

        logger.info(json.dumps({"event": "training_complete", "model_path": pipeline_path}))
        print("Training complete. Pipeline saved to:", pipeline_path)
        return pipeline_path

    except Exception as e:
        logger.exception(json.dumps({"event": "training_error", "error": str(e)}))
        raise


def parse_args_and_run():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/raw/dataset.csv", help="Path to dataset (DVC-tracked)")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--problem_type", required=True, choices=["classification", "regression"])
    p.add_argument("--model_dir", default="api/model", help="Model output directory")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--poison", default=None, help="Optional poison spec for simulation (e.g. 'flip:0.1')")
    p.add_argument("--tune", action="store_true", help="Run quick hyperparameter tuning (small search)")
    args = p.parse_args()

    train_main(
        data_path=args.data_path,
        target=args.target,
        problem_type=args.problem_type,
        model_dir=args.model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        poison=args.poison,
        tune=args.tune,
    )


if __name__ == "__main__":
    parse_args_and_run()

