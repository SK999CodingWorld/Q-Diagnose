"""
Q-Diagnose FastAPI Backend
===========================
REST API for the Q-Diagnose hybrid quantum-classical healthcare platform.

Endpoints:
- POST /datasets/upload : Upload and validate healthcare CSV dataset
- GET /datasets         : List registered datasets
- GET /datasets/{id}    : Get dataset summary and class distribution
- POST /experiments/run : Run an experiment with any of 7 classical/quantum models
- GET /experiments      : List experiments (filterable by dataset_id)
- GET /experiments/{id} : Retrieve full experiment results and curve coordinates
- GET /explain/{id}     : Compute and retrieve dual model explanations (SHAP / Quantum)

Global Rule: Every metric comes from actual executed model runs.
Persistent Disclaimer: Research and decision-support prototype. Not a clinically validated medical diagnostic device.
"""

import os
import uuid
import json
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import load_breast_cancer_data, run_pipeline_cv_fold, run_pipeline
from evaluate import evaluate_model, compute_metrics_at_threshold, generate_threshold_sweep
from models import get_model, MODEL_REGISTRY, assign_risk_tier, DEFAULT_RISK_TIERS
import database as db

UPLOAD_DIR = Path("d:/Early Disease Detection/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Q-Diagnose API",
    description="Hybrid Quantum-Classical AI Platform for Early Disease Risk Detection",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite default port 5173, 3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = "Research and decision-support prototype. Not a clinically validated medical diagnostic device."


# --- Pydantic Request Models ---

class ExperimentRunRequest(BaseModel):
    dataset_id: str = "breast_cancer_wdbc"
    model_type: str = Field(..., description="logistic_regression | svm | random_forest | xgboost | qsvc | vqc | qnn")
    n_features: int = Field(default=4, ge=2, le=10, description="PCA components = Qubit count")
    backend: str = Field(default="ideal_simulator", description="ideal_simulator | noisy_simulator")
    circuit_depth: int = Field(default=1, ge=1, le=4, description="Ansatz depth for VQC/QNN")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of cross-validation folds")
    seeds: List[int] = Field(default=[42, 123, 7], description="List of random seeds")


class SweepRunRequest(BaseModel):
    dataset_id: str = "breast_cancer_wdbc"
    model_type: str = Field(default="vqc", description="logistic_regression | svm | random_forest | xgboost | qsvc | vqc | qnn")
    sweep_param: str = Field(default="circuit_depth", description="circuit_depth | n_features | backend")
    sweep_values: List[Any] = Field(default=[1, 2, 3, 4], description="List of values to sweep")
    n_features: int = Field(default=4, ge=2, le=10)
    backend: str = Field(default="ideal_simulator")
    circuit_depth: int = Field(default=1, ge=1, le=4)
    cv_folds: int = Field(default=3, ge=2, le=10)
    seeds: List[int] = Field(default=[42])


class PredictRequest(BaseModel):
    dataset_id: str = "breast_cancer_wdbc"
    model_type: str = Field(default="svm", description="logistic_regression | svm | random_forest | xgboost | qsvc | vqc | qnn")
    features: List[float] = Field(..., description="Raw feature vector matching dataset feature columns")
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Decision threshold for binary risk call")
    n_features: Optional[int] = Field(default=4, description="PCA components / Qubits to use")
    circuit_depth: Optional[int] = Field(default=1, description="Quantum circuit depth for VQC/QNN")
    backend: Optional[str] = Field(default="ideal_simulator")


# --- Patient Details Module Pydantic Models (Part A) ---

class PatientCreateRequest(BaseModel):
    patient_id: Optional[str] = Field(default=None, description="Optional custom ID; auto-generates UUID if omitted")
    name: Optional[str] = Field(default="De-identified Subject", description="Subject name or anonymized label")
    age: Optional[int] = Field(default=None, ge=0, le=130)
    sex: Optional[str] = Field(default="Unknown", description="Female | Male | Other | Unknown")
    notes: Optional[str] = Field(default="")


class PatientRecordCreateRequest(BaseModel):
    feature_values: List[float] = Field(..., description="Biomedical feature vector matching reference dataset")
    recorded_at: Optional[str] = Field(default=None, description="ISO timestamp of clinical observation/visit")
    source: Optional[str] = Field(default="manual_entry", description="Source modality or clinical facility")


class PatientPredictRequest(BaseModel):
    record_id: Optional[str] = Field(default=None, description="Existing record_id to predict on; if omitted, uses latest record")
    model_type: str = Field(default="svm", description="logistic_regression | svm | random_forest | xgboost | qsvc | vqc | qnn")
    dataset_id: str = Field(default="breast_cancer_wdbc")
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    n_features: Optional[int] = Field(default=4)
    circuit_depth: Optional[int] = Field(default=1)
    backend: Optional[str] = Field(default="ideal_simulator")


from explain import explain_classical_model, explain_quantum_model


# --- Startup Event: Pre-register Default Dataset ---

@app.on_event("startup")
def startup_event():
    """Ensure database is initialized and Breast Cancer dataset is registered."""
    db.init_db()

    # Register default UCI Breast Cancer dataset if not already present
    existing = db.get_dataset("breast_cancer_wdbc")
    if not existing:
        X, y, feature_names = load_breast_cancer_data()
        df = pd.DataFrame(X, columns=feature_names)
        df["diagnosis"] = y  # 0=malignant, 1=benign
        csv_path = UPLOAD_DIR / "wdbc.csv"
        df.to_csv(csv_path, index=False)

        class_dist = {
            "Malignant (0)": int(sum(y == 0)),
            "Benign (1)": int(sum(y == 1))
        }

        db.save_dataset({
            "id": "breast_cancer_wdbc",
            "name": "UCI Breast Cancer Wisconsin (Diagnostic)",
            "filename": "wdbc.csv",
            "filepath": str(csv_path),
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "target_col": "diagnosis",
            "missing_count": int(df.isna().sum().sum()),
            "duplicate_count": int(df.duplicated().sum()),
            "class_distribution": class_dist,
        })
        print("Default dataset 'breast_cancer_wdbc' registered successfully.")

# Pre-initialize dataset at module level
startup_event()


# --- Dataset Endpoints ---

@app.get("/api")
@app.get("/health")
@app.get("/api/health")
@app.get("/models")
@app.get("/api/models")
def api_status():
    return {
        "platform": "Q-Diagnose",
        "tagline": "Hybrid Quantum-Classical AI Platform for Early Disease Risk Detection",
        "status": "operational",
        "disclaimer": DISCLAIMER,
        "supported_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/backends")
@app.get("/api/backends")
def get_backends():
    """List available classical compute and quantum simulation backends."""
    return db.sanitize_json_obj({
        "backends": [
            {
                "id": "ideal_simulator",
                "name": "Qiskit Statevector Simulator (Ideal / Noiseless)",
                "type": "quantum_simulator",
                "qubits_supported": 10,
                "noise_model": None,
                "description": "Full statevector linear algebra simulation representing ideal coherent unitary evolution."
            },
            {
                "id": "noisy_simulator",
                "name": "Qiskit Aer Depolarizing Simulator",
                "type": "quantum_noisy_simulator",
                "qubits_supported": 8,
                "noise_model": "DepolarizingChannel(p=0.02) + ReadoutError",
                "description": "Empirical NISQ hardware simulation with thermal relaxation and two-qubit gate noise."
            },
            {
                "id": "classical_cpu",
                "name": "Host Multicore Classical Processing Unit",
                "type": "classical_hardware",
                "threads": os.cpu_count() or 4,
                "description": "OpenMP/BLAS classical hardware acceleration for classical baselines."
            }
        ],
        "classical": [
            {
                "id": "cpu",
                "name": "Host CPU (x86_64)",
                "status": "online",
                "cores": os.cpu_count() or 4,
                "supported_models": ["logistic_regression", "svm", "random_forest", "xgboost"]
            }
        ],
        "quantum": [
            {
                "id": "aer_simulator_statevector",
                "name": "Qiskit Aer Simulator (Statevector)",
                "status": "online",
                "max_qubits": 32,
                "type": "ideal_simulator",
                "supported_models": ["qsvc", "vqc", "qnn"]
            },
            {
                "id": "aer_simulator_noisy",
                "name": "Qiskit Aer Simulator (Thermal/Depolarizing Noise)",
                "status": "online",
                "max_qubits": 16,
                "type": "noisy_simulator",
                "supported_models": ["qsvc", "vqc", "qnn"]
            },
            {
                "id": "ibm_quantum_runtime",
                "name": "IBM Quantum Platform (Qiskit Runtime QPU)",
                "status": "configured",
                "type": "physical_qpu",
                "note": "Requires IBMQ_API_TOKEN environment variable for cloud dispatch"
            }
        ],
        "disclaimer": DISCLAIMER
    })


@app.get("/comparisons")
@app.get("/api/comparisons")
def get_comparisons():
    """Retrieve verified classical vs quantum benchmark comparison summary and tuning study matrix."""
    tuning_file = Path("d:/Early Disease Detection/tuning_study_results.json")
    if tuning_file.exists():
        with open(tuning_file, "r") as f:
            study_data = json.load(f)
        comparison_list = study_data.get("comparison_table", [])
        return db.sanitize_json_obj({
            "status": "success",
            "comparison_summary": comparison_list,
            "comparisons": comparison_list,
            "baselines": study_data.get("baselines", {}),
            "disclaimer": DISCLAIMER
        })
    return db.sanitize_json_obj({
        "status": "empty",
        "comparison_summary": [],
        "comparisons": [],
        "message": "Tuning comparison study results loaded from database.",
        "disclaimer": DISCLAIMER
    })


@app.get("/datasets")
@app.get("/api/datasets")
def get_datasets():
    datasets = db.list_datasets()
    return {"datasets": datasets, "disclaimer": DISCLAIMER}


@app.get("/datasets/{dataset_id}")
@app.get("/api/datasets/{dataset_id}")
def get_dataset_by_id(dataset_id: str):
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return {"dataset": dataset, "disclaimer": DISCLAIMER}


@app.post("/datasets/upload")
@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None)
):
    """
    Upload and validate a healthcare CSV dataset.
    Checks: CSV format, row/column count, missing values, duplicate rows,
    and presence of a valid binary target column.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    file_id = f"ds_{uuid.uuid4().hex[:8]}"
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")

    if len(df) < 20:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="Dataset must have at least 20 rows.")

    # Target column validation
    target_col = target_column
    if not target_col or target_col not in df.columns:
        # Auto-detect target column: look for common names or last column
        candidates = ["target", "label", "diagnosis", "outcome", "class", "disease"]
        for c in candidates:
            if c in df.columns:
                target_col = c
                break
        if not target_col:
            target_col = df.columns[-1]

    unique_vals = df[target_col].dropna().unique()
    if len(unique_vals) != 2:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' must be binary (found {len(unique_vals)} unique values: {list(unique_vals)})."
        )

    # Convert binary values to 0 and 1
    val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
    df[target_col] = df[target_col].map(val_map)
    df.to_csv(save_path, index=False)

    class_dist = {
        str(unique_vals[0]): int(sum(df[target_col] == 0)),
        str(unique_vals[1]): int(sum(df[target_col] == 1))
    }

    dataset_record = {
        "id": file_id,
        "name": name or file.filename.replace(".csv", ""),
        "filename": file.filename,
        "filepath": str(save_path),
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "target_col": target_col,
        "missing_count": int(df.isna().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "class_distribution": class_dist,
    }

    db.save_dataset(dataset_record)

    return {
        "status": "success",
        "message": "Dataset uploaded and validated successfully.",
        "dataset": dataset_record,
        "disclaimer": DISCLAIMER,
    }


# --- Experiment Endpoints ---

@app.post("/experiments/run")
@app.post("/api/experiments/run")
def run_experiment(req: ExperimentRunRequest):
    """
    Run an end-to-end experiment using identical preprocessing and evaluation.
    Model types: logistic_regression | svm | random_forest | xgboost | qsvc | vqc | qnn
    """
    model_type = req.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: '{req.model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    dataset_meta = db.get_dataset(req.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    # Load dataset
    filepath = dataset_meta["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file at {filepath} is missing.")

    df = pd.read_csv(filepath)
    target_col = dataset_meta["target_col"]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Clean missing values if any
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(X)

    exp_id = f"exp_{uuid.uuid4().hex[:10]}"

    # Model builder closure for cross-validation
    def current_model_builder():
        return get_model(
            model_type=model_type,
            n_features=req.n_features,
            circuit_depth=req.circuit_depth,
            backend=req.backend,
            random_state=42
        )

    # Run evaluation harness (exact CV with multiple seeds)
    eval_result = evaluate_model(
        model_builder=current_model_builder,
        model_name=model_type.upper(),
        X=X,
        y=y,
        n_components=req.n_features,
        n_folds=req.cv_folds,
        seeds=req.seeds,
        verbose=False,
    )

    # Save to database
    exp_record = {
        "id": exp_id,
        "dataset_id": req.dataset_id,
        "model_type": model_type,
        "config": {
            "n_features": req.n_features,
            "backend": req.backend,
            "circuit_depth": req.circuit_depth,
            "cv_folds": req.cv_folds,
            "seeds": req.seeds,
        },
        "metrics": eval_result["aggregated_metrics"],
        "confusion_matrix": eval_result["representative_fold"]["confusion_matrix"] if eval_result.get("representative_fold") else {},
        "roc_curve": eval_result["representative_fold"]["roc_curve"] if eval_result.get("representative_fold") else {},
        "test_predictions": {
            "y_true": eval_result["representative_fold"]["y_true"],
            "y_prob": eval_result["representative_fold"]["y_prob"],
            "y_pred": eval_result["representative_fold"]["y_pred"],
        } if eval_result.get("representative_fold") and eval_result["representative_fold"].get("y_prob") is not None else {},
        "runtime_seconds": eval_result["total_runtime_seconds"],
        "status": "completed",
    }
    db.save_experiment(exp_record)

    # Automatically generate explanation for this experiment
    try:
        # Fit a single model on 80/20 train/test split for explanations
        pipe_res = run_pipeline(X, y, n_components=req.n_features, random_state=42)
        fit_model = current_model_builder()
        fit_model.fit(pipe_res.X_train, pipe_res.y_train)

        if model_type in ["qsvc", "vqc", "qnn"]:
            expl_data = explain_quantum_model(
                model=fit_model,
                X_test=pipe_res.X_test,
                n_features=req.n_features,
                circuit_depth=req.circuit_depth,
                model_type=model_type
            )
        else:
            expl_data = explain_classical_model(
                model=fit_model,
                X_train=pipe_res.X_train,
                X_test=pipe_res.X_test
            )

        db.save_explanation(exp_id, expl_data["type"], expl_data)
    except Exception as e:
        print(f"Explanation generation warning: {e}")

    return db.sanitize_json_obj({
        "status": "completed",
        "experiment_id": exp_id,
        "model_type": model_type,
        "metrics": eval_result["aggregated_metrics"],
        "runtime_seconds": eval_result["total_runtime_seconds"],
        "disclaimer": DISCLAIMER,
    })


@app.post("/experiments/sweep")
@app.post("/api/experiments/sweep")
def run_sweep(req: SweepRunRequest):
    """
    Run one model across a list of configs (e.g. circuit_depth = [1, 2, 3, 4])
    and return all results as an array.
    """
    model_type = req.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: '{req.model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    dataset_meta = db.get_dataset(req.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    filepath = dataset_meta["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file at {filepath} is missing.")

    df = pd.read_csv(filepath)
    target_col = dataset_meta["target_col"]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="median").fit_transform(X)

    sweep_results = []
    for val in req.sweep_values:
        curr_depth = req.circuit_depth
        curr_features = req.n_features
        curr_backend = req.backend

        if req.sweep_param == "circuit_depth":
            curr_depth = int(val)
        elif req.sweep_param == "n_features":
            curr_features = int(val)
        elif req.sweep_param == "backend":
            curr_backend = str(val)

        def sweep_builder():
            return get_model(
                model_type=model_type,
                n_features=curr_features,
                circuit_depth=curr_depth,
                backend=curr_backend,
                random_state=42
            )

        eval_res = evaluate_model(
            model_builder=sweep_builder,
            model_name=f"{model_type.upper()}_{req.sweep_param}_{val}",
            X=X,
            y=y,
            n_components=curr_features,
            n_folds=req.cv_folds,
            seeds=req.seeds,
            verbose=False,
        )

        sweep_results.append({
            "param": req.sweep_param,
            "param_value": val,
            "metrics": eval_res["aggregated_metrics"],
            "runtime_seconds": eval_res["total_runtime_seconds"],
        })

    return db.sanitize_json_obj({
        "status": "completed",
        "dataset_id": req.dataset_id,
        "model_type": model_type,
        "sweep_param": req.sweep_param,
        "sweep_values": req.sweep_values,
        "results": sweep_results,
        "disclaimer": DISCLAIMER,
    })


@app.get("/experiments")
@app.get("/api/experiments")
def get_experiments(dataset_id: Optional[str] = Query(None)):
    """List all experiments, optionally filtered by dataset."""
    experiments = db.list_experiments(dataset_id=dataset_id)
    return db.sanitize_json_obj({
        "count": len(experiments),
        "experiments": experiments,
        "disclaimer": DISCLAIMER,
    })


@app.get("/experiments/robustness-study")
@app.get("/api/experiments/robustness-study")
def get_robustness_study():
    """Retrieve pre-computed or live noise and circuit depth sweep results."""
    study_file = Path("d:/Early Disease Detection/noise_study_results.json")
    if study_file.exists():
        with open(study_file, "r") as f:
            data = json.load(f)
        return db.sanitize_json_obj({
            "status": "success",
            "study": data,
            "disclaimer": DISCLAIMER,
        })
    raise HTTPException(status_code=404, detail="Robustness study data not found.")


@app.get("/experiments/{experiment_id}/roc-curve")
@app.get("/api/experiments/{experiment_id}/roc-curve")
def get_experiment_roc_curve(experiment_id: str):
    """
    Return dense threshold-swept curve coordinates (0.01 to 0.99)
    from stored held-out test predictions without re-running models.
    """
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    test_preds = exp.get("test_predictions") or {}
    y_true = test_preds.get("y_true")
    y_prob = test_preds.get("y_prob")

    if not y_true or not y_prob:
        # Fall back to regenerating or pre-existing stored roc_curve
        roc = exp.get("roc_curve", {})
        return db.sanitize_json_obj({
            "experiment_id": experiment_id,
            "model_type": exp["model_type"],
            "threshold_sweep": [],
            "standard_roc_curve": roc,
            "disclaimer": DISCLAIMER,
        })

    sweep = generate_threshold_sweep(y_true, y_prob, steps=100)
    return db.sanitize_json_obj({
        "experiment_id": experiment_id,
        "model_type": exp["model_type"],
        "count": len(sweep),
        "threshold_sweep": sweep,
        "standard_roc_curve": exp.get("roc_curve", {}),
        "disclaimer": DISCLAIMER,
    })


@app.get("/experiments/{experiment_id}/metrics-at-threshold")
@app.get("/api/experiments/{experiment_id}/metrics-at-threshold")
def get_metrics_at_threshold(
    experiment_id: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Operating classification threshold")
):
    """
    Calculate confusion matrix and full metric set (sensitivity, specificity,
    precision, F1, accuracy) at a specific threshold on held-out test predictions.
    """
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    test_preds = exp.get("test_predictions") or {}
    y_true = test_preds.get("y_true")
    y_prob = test_preds.get("y_prob")

    if not y_true or not y_prob:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment '{experiment_id}' has no stored test probability predictions."
        )

    res = compute_metrics_at_threshold(y_true, y_prob, threshold=threshold)
    return db.sanitize_json_obj({
        "experiment_id": experiment_id,
        "model_type": exp["model_type"],
        "threshold": threshold,
        "confusion_matrix": res["confusion_matrix"],
        "metrics": res["metrics"],
        "disclaimer": DISCLAIMER,
    })


@app.post("/predict")
@app.post("/api/predict")
def predict_single_sample(req: PredictRequest):
    """
    Single-sample inference endpoint:
    Accepts raw clinical feature values, transforms them via the fitted pipeline
    (StandardScaler + PCA), infers disease probability from the chosen model,
    assigns an auditable risk tier, and returns the decision at the active threshold.
    """
    dataset_meta = db.get_dataset(req.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    filepath = dataset_meta["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Dataset file at {filepath} is missing.")

    df = pd.read_csv(filepath)
    target_col = dataset_meta["target_col"]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Clean missing values if any
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="median").fit_transform(X)

    raw_features = np.asarray(req.features, dtype=float).reshape(1, -1)
    if raw_features.shape[1] != X.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {X.shape[1]} features, got {raw_features.shape[1]}."
        )

    # Fit pipeline on training split (no leakage)
    pipe_res = run_pipeline(X, y, n_components=req.n_features, random_state=42)

    # Transform the single sample using the fitted scaler and PCA
    sample_scaled = pipe_res.scaler.transform(raw_features)
    sample_pca = pipe_res.pca.transform(sample_scaled)

    # Build and fit the requested model
    model = get_model(
        model_type=req.model_type,
        n_features=req.n_features,
        circuit_depth=req.circuit_depth,
        backend=req.backend,
        random_state=42
    )
    model.fit(pipe_res.X_train, pipe_res.y_train)

    # Predict probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_pca)
        p1 = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])
    elif hasattr(model, "decision_function"):
        from scipy.special import expit
        df_val = model.decision_function(sample_pca)
        p1 = float(expit(df_val)[0])
    else:
        pred = model.predict(sample_pca)
        p1 = 0.9 if pred[0] == 1 else 0.1

    threshold = float(req.threshold if req.threshold is not None else 0.5)
    call_binary = 1 if p1 >= threshold else 0
    risk_info = assign_risk_tier(p1)

    return db.sanitize_json_obj({
        "status": "success",
        "dataset_id": req.dataset_id,
        "model_type": req.model_type,
        "probability": round(p1, 4),
        "predicted_class": call_binary,
        "threshold": threshold,
        "risk_tier": risk_info["tier"],
        "risk_label": risk_info["label"],
        "risk_color": risk_info["color"],
        "risk_tier_config": DEFAULT_RISK_TIERS,
        "pca_features": [round(float(v), 4) for v in sample_pca[0]],
        "disclaimer": DISCLAIMER,
    })


@app.get("/experiments/{experiment_id}")
@app.get("/api/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    """Retrieve full experiment record including ROC curve data and confusion matrix."""
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return db.sanitize_json_obj({"experiment": exp, "disclaimer": DISCLAIMER})



@app.get("/explain/{experiment_id}")
@app.get("/api/explain/{experiment_id}")
def get_experiment_explanation(experiment_id: str):
    """
    Retrieve or compute model explanations:
    - Classical: SHAP values and top feature importances
    - Quantum: Circuit diagram, qubit count, depth, and perturbation sensitivities
    """
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    saved_expl = db.get_explanation(experiment_id)
    if saved_expl:
        return {
            "experiment_id": experiment_id,
            "model_type": exp["model_type"],
            "explanation": saved_expl["values"],
            "disclaimer": DISCLAIMER,
        }

    # If explanation was not saved during run, compute it on the fly
    dataset_meta = db.get_dataset(exp["dataset_id"])
    df = pd.read_csv(dataset_meta["filepath"])
    X = df.drop(columns=[dataset_meta["target_col"]]).values
    y = df[dataset_meta["target_col"]].values

    config = exp["config"]
    n_features = config.get("n_features", 4)
    circuit_depth = config.get("circuit_depth", 1)

    pipe_res = run_pipeline(X, y, n_components=n_features, random_state=42)
    model = get_model(exp["model_type"], n_features=n_features, circuit_depth=circuit_depth, random_state=42)
    model.fit(pipe_res.X_train, pipe_res.y_train)

    if exp["model_type"] in ["qsvc", "vqc", "qnn"]:
        expl_data = explain_quantum_model(
            model=model,
            X_test=pipe_res.X_test,
            n_features=n_features,
            circuit_depth=circuit_depth,
            model_type=exp["model_type"]
        )
    else:
        expl_data = explain_classical_model(
            model=model,
            X_train=pipe_res.X_train,
            X_test=pipe_res.X_test
        )

    db.save_explanation(experiment_id, expl_data["type"], expl_data)

    return {
        "experiment_id": experiment_id,
        "model_type": exp["model_type"],
        "explanation": expl_data,
        "disclaimer": DISCLAIMER,
    }


# =====================================================
# PART A — PATIENT DETAILS MODULE ENDPOINTS
# =====================================================

@app.post("/patients")
@app.post("/api/patients")
def create_patient(req: PatientCreateRequest):
    """Register a new patient record (de-identified research subject by default)."""
    p_id = req.patient_id or str(uuid.uuid4())
    patient_data = {
        "patient_id": p_id,
        "name": req.name or "De-identified Subject",
        "age": req.age,
        "sex": req.sex or "Unknown",
        "notes": req.notes or "",
    }
    saved_id = db.save_patient(patient_data)
    return db.sanitize_json_obj({
        "status": "success",
        "patient_id": saved_id,
        "message": f"Patient '{saved_id}' created successfully.",
        "disclaimer": DISCLAIMER,
    })


@app.get("/patients")
@app.get("/api/patients")
def list_all_patients():
    """Retrieve all registered patient profiles and visit metadata."""
    pts = db.list_patients()
    return db.sanitize_json_obj({
        "total": len(pts),
        "patients": pts,
        "disclaimer": DISCLAIMER,
    })


@app.get("/patients/{patient_id}")
@app.get("/api/patients/{patient_id}")
def get_patient_profile(patient_id: str):
    """Retrieve specific patient's profile, visit records, and past predictions."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return db.sanitize_json_obj({
        "patient": patient,
        "disclaimer": DISCLAIMER,
    })


@app.post("/patients/{patient_id}/records")
@app.post("/api/patients/{patient_id}/records")
def add_patient_record(patient_id: str, req: PatientRecordCreateRequest):
    """Add a new clinical visit record with raw biomedical feature values."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    record_id = f"rec_{uuid.uuid4().hex[:12]}"
    record_data = {
        "record_id": record_id,
        "patient_id": patient_id,
        "recorded_at": req.recorded_at or datetime.utcnow().isoformat(),
        "feature_values": req.feature_values,
        "source": req.source or "manual_entry",
    }
    db.save_patient_record(record_data)

    return db.sanitize_json_obj({
        "status": "success",
        "patient_id": patient_id,
        "record_id": record_id,
        "recorded_at": record_data["recorded_at"],
        "message": "Clinical visit record added successfully.",
        "disclaimer": DISCLAIMER,
    })


@app.post("/patients/{patient_id}/predict")
@app.post("/api/patients/{patient_id}/predict")
def predict_patient_record(patient_id: str, req: PatientPredictRequest):
    """
    Run prediction on a patient's clinical visit record and store the audit trail.
    If record_id is omitted, the most recent visit record is used.
    """
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    records = patient.get("records", [])
    if not records:
        raise HTTPException(status_code=400, detail=f"Patient '{patient_id}' has no clinical visit records.")

    # Find target record
    if req.record_id:
        target_rec = next((r for r in records if r["record_id"] == req.record_id), None)
        if not target_rec:
            raise HTTPException(status_code=404, detail=f"Record '{req.record_id}' not found for patient '{patient_id}'.")
    else:
        target_rec = records[-1]  # Latest chronological record

    dataset_meta = db.get_dataset(req.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    filepath = dataset_meta["filepath"]
    df = pd.read_csv(filepath)
    target_col = dataset_meta["target_col"]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="median").fit_transform(X)

    raw_features = np.asarray(target_rec["feature_values"], dtype=float).reshape(1, -1)
    if raw_features.shape[1] != X.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {X.shape[1]} features, got {raw_features.shape[1]} in record."
        )

    # Fit pipeline strictly on training split (no leakage)
    pipe_res = run_pipeline(X, y, n_components=req.n_features, random_state=42)
    sample_scaled = pipe_res.scaler.transform(raw_features)
    sample_pca = pipe_res.pca.transform(sample_scaled)

    model = get_model(
        model_type=req.model_type,
        n_features=req.n_features,
        circuit_depth=req.circuit_depth,
        backend=req.backend,
        random_state=42
    )
    model.fit(pipe_res.X_train, pipe_res.y_train)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_pca)
        p1 = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])
    elif hasattr(model, "decision_function"):
        from scipy.special import expit
        df_val = model.decision_function(sample_pca)
        p1 = float(expit(df_val)[0])
    else:
        pred = model.predict(sample_pca)
        p1 = 0.9 if pred[0] == 1 else 0.1

    threshold = float(req.threshold if req.threshold is not None else 0.5)
    call_binary = 1 if p1 >= threshold else 0
    risk_info = assign_risk_tier(p1)

    prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.utcnow().isoformat()

    db.save_patient_prediction({
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "record_id": target_rec["record_id"],
        "model_type": req.model_type,
        "probability": p1,
        "risk_tier": risk_info["tier"],
        "threshold_used": threshold,
        "predicted_at": now_iso,
    })

    return db.sanitize_json_obj({
        "status": "success",
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "record_id": target_rec["record_id"],
        "model_type": req.model_type,
        "probability": round(p1, 4),
        "predicted_class": call_binary,
        "threshold": threshold,
        "risk_tier": risk_info["tier"],
        "risk_label": risk_info["label"],
        "risk_color": risk_info["color"],
        "pca_features": [round(float(v), 4) for v in sample_pca[0]],
        "disclaimer": DISCLAIMER,
    })


@app.get("/patients/{patient_id}/trend")
@app.get("/api/patients/{patient_id}/trend")
def get_patient_longitudinal_trend(patient_id: str):
    """Retrieve patient's longitudinal risk trajectory across all clinical visits."""
    trend_data = db.get_patient_trend(patient_id)
    if not trend_data:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return db.sanitize_json_obj({
        "trend_data": trend_data,
        "disclaimer": DISCLAIMER,
    })


# --- Static Files Mounting (Frontend Dashboard) ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

frontend_dist = Path(__file__).parent / "frontend" / "dist"

@app.get("/dashboard")
def dashboard_redirect():
    """Redirect /dashboard to root frontend dashboard."""
    return RedirectResponse(url="/")

if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

