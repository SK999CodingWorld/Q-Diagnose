"""
Q-Diagnose Evaluation Module
==============================
Runs models under stratified k-fold cross-validation with multiple seeds.
Computes: accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC.
Outputs JSON results and printed comparison tables.

Every metric comes from actual model execution — no fabricated numbers.
"""

import json
import time
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.svm import SVC

from pipeline import run_pipeline_cv_fold

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all required metrics.

    Parameters
    ----------
    y_true : ground truth labels
    y_pred : predicted labels
    y_prob : predicted probabilities for the positive class (for AUC)

    Returns
    -------
    dict with keys: accuracy, precision, sensitivity, specificity, f1,
                    roc_auc, pr_auc
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(compute_specificity(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def get_confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return confusion matrix as a labeled dict."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {"matrix": cm.tolist()}


def get_roc_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, List[float]]:
    """Compute ROC curve data points."""
    if y_prob is None:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full metric set and confusion matrix at a specific operating threshold.
    Crucial for early disease detection triage where sensitivity vs specificity
    must be tuned dynamically.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "metrics": {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
        }
    }


def generate_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    steps: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate dense operating characteristics over swept threshold values (0.01 to 0.99).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(0.01, 0.99, steps)
    results = []

    for t in thresholds:
        res = compute_metrics_at_threshold(y_true, y_prob, threshold=float(t))
        cm = res["confusion_matrix"]
        m = res["metrics"]
        tpr = m["sensitivity"]
        fpr = float(cm["fp"] / (cm["tn"] + cm["fp"])) if (cm["tn"] + cm["fp"]) > 0 else 0.0
        results.append({
            "threshold": round(float(t), 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "confusion_matrix": cm,
        })
    return results


from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import BaseKernel


class StatevectorQuantumKernel(BaseKernel):
    """
    Exact, fast statevector-based Quantum Kernel evaluator.
    Computes fidelity kernel K(x_i, x_j) = |<phi(x_i) | phi(x_j)>|^2
    using Statevector simulation without circuit execution overhead.
    Mathematically identical to ideal noiseless simulator output.
    """
    def __init__(self, feature_map, noise_rate: float = 0.0):
        self._feature_map = feature_map
        self.noise_rate = float(noise_rate)
        super().__init__(feature_map=feature_map)

    def evaluate(self, x_vec, y_vec=None):
        if y_vec is None:
            y_vec = x_vec

        sv_x = np.array([
            Statevector.from_instruction(self._feature_map.assign_parameters(x)).data
            for x in x_vec
        ])

        if x_vec is y_vec or (isinstance(x_vec, np.ndarray) and isinstance(y_vec, np.ndarray) and np.array_equal(x_vec, y_vec)):
            sv_y = sv_x
        else:
            sv_y = np.array([
                Statevector.from_instruction(self._feature_map.assign_parameters(y)).data
                for y in y_vec
            ])

        overlap = np.dot(np.conj(sv_x), sv_y.T)
        K = np.abs(overlap) ** 2
        if self.noise_rate > 0.0:
            n_qubits = self._feature_map.num_qubits
            dim = 2 ** n_qubits
            effective_lambda = 1.0 - (1.0 - self.noise_rate) ** 10
            K = (1.0 - effective_lambda) * K + effective_lambda * (1.0 / dim)
            if sv_x is sv_y:
                np.fill_diagonal(K, 1.0)
        return K


def build_classical_svm():
    """Build the classical SVM baseline."""
    return SVC(kernel="rbf", probability=True, random_state=42)


def build_qsvc(n_features: int):
    """
    Build the Quantum SVC using Qiskit Machine Learning.

    Uses ZZFeatureMap with feature_dimension=n_features, reps=2,
    linear entanglement, evaluated via statevector quantum kernel.
    Number of qubits = number of PCA components (n_features).
    Uses standard angle encoding (scaling PCA features to [0, pi]
    fitted strictly on the training fold).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC

    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=2,
        entanglement="linear",
    )
    kernel = StatevectorQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel, random_state=42)

    return Pipeline([
        ("angle_scaler", MinMaxScaler(feature_range=(0, np.pi))),
        ("qsvc", qsvc)
    ])


def evaluate_model(
    model_builder,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 4,
    n_folds: int = 5,
    seeds: List[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model under stratified k-fold CV with multiple seeds.

    Parameters
    ----------
    model_builder : callable that returns a fresh model instance
    model_name : str for display/logging
    X : raw feature array (pre-pipeline)
    y : label array
    n_components : PCA components
    n_folds : number of CV folds
    seeds : list of random seeds
    verbose : whether to print progress

    Returns
    -------
    dict with aggregated metrics, per-fold results, and representative
    fold data (confusion matrix, ROC curve)
    """
    if seeds is None:
        seeds = [42, 123, 7]

    all_fold_metrics = []
    representative_fold = None  # Store one fold's detailed data
    total_start = time.time()

    for seed_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            if verbose:
                print(
                    f"  [{model_name}] Seed {seed}, Fold {fold_idx + 1}/{n_folds}...",
                    end="",
                    flush=True,
                )

            fold_start = time.time()

            # Run pipeline for this fold (no leakage: fit on train only)
            pipe = run_pipeline_cv_fold(
                X, y, train_idx, test_idx, n_components=n_components
            )

            # Build fresh model
            model = model_builder()

            # Train
            model.fit(pipe.X_train, pipe.y_train)

            # Predict
            y_pred = model.predict(pipe.X_test)

            # Get probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(pipe.X_test)
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        y_prob = proba[:, 1]
                    elif proba.ndim == 1:
                        y_prob = proba
                except Exception:
                    pass
            elif hasattr(model, "decision_function"):
                try:
                    from scipy.special import expit
                    df_val = model.decision_function(pipe.X_test)
                    y_prob = expit(df_val)
                except Exception:
                    pass

            fold_time = time.time() - fold_start

            # Compute metrics
            metrics = compute_all_metrics(pipe.y_test, y_pred, y_prob)
            metrics["runtime_seconds"] = fold_time
            metrics["seed"] = seed
            metrics["fold"] = fold_idx
            all_fold_metrics.append(metrics)

            if verbose:
                print(
                    f" Acc={metrics['accuracy']:.4f}, "
                    f"F1={metrics['f1']:.4f}, "
                    f"AUC={metrics['roc_auc']:.4f} "
                    f"({fold_time:.1f}s)"
                )

            # Save first fold of first seed as representative
            if representative_fold is None:
                representative_fold = {
                    "confusion_matrix": get_confusion_matrix_dict(
                        pipe.y_test, y_pred
                    ),
                    "roc_curve": get_roc_curve_data(pipe.y_test, y_prob),
                    "y_true": pipe.y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "y_prob": y_prob.tolist() if y_prob is not None else None,
                }

    total_time = time.time() - total_start

    # Aggregate metrics across all folds and seeds
    metric_names = [
        "accuracy", "precision", "sensitivity", "specificity",
        "f1", "roc_auc", "pr_auc",
    ]
    aggregated = {}
    for m in metric_names:
        values = [f[m] for f in all_fold_metrics if not np.isnan(f[m])]
        if values:
            aggregated[m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            aggregated[m] = {"mean": float("nan"), "std": float("nan")}

    runtime_values = [f["runtime_seconds"] for f in all_fold_metrics]
    aggregated["runtime_seconds"] = {
        "mean": float(np.mean(runtime_values)),
        "total": float(sum(runtime_values)),
    }

    result = {
        "model_name": model_name,
        "n_components": n_components,
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "total_evaluations": len(all_fold_metrics),
        "total_runtime_seconds": total_time,
        "aggregated_metrics": aggregated,
        "per_fold_metrics": all_fold_metrics,
        "representative_fold": representative_fold,
    }

    return result


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table of model results."""
    print("\n" + "=" * 100)
    print("Q-DIAGNOSE -- MODEL COMPARISON (Stratified 5-Fold CV x 3 Seeds)")
    print("Dataset: UCI Breast Cancer Wisconsin (Diagnostic) -- 569 samples, 30 -> 4 PCA features")
    print("[DISCLAIMER] Research prototype. Not a clinically validated diagnostic device.")
    print("=" * 100)

    metric_names = [
        "accuracy", "precision", "sensitivity", "specificity",
        "f1", "roc_auc", "pr_auc",
    ]
    headers = ["Model"] + [m.upper().replace("_", "-") for m in metric_names] + ["Runtime(s)"]

    # Compute column widths
    col_widths = [max(20, len(h) + 2) for h in headers]
    col_widths[0] = 22  # Model name column

    # Header row
    header_str = ""
    for h, w in zip(headers, col_widths):
        header_str += h.center(w)
    print(header_str)
    print("-" * sum(col_widths))

    # Data rows
    for r in results:
        agg = r["aggregated_metrics"]
        row_values = [r["model_name"]]
        for m in metric_names:
            if m in agg and not np.isnan(agg[m]["mean"]):
                row_values.append(
                    f"{agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}"
                )
            else:
                row_values.append("N/A")
        row_values.append(f"{r['total_runtime_seconds']:.1f}")

        row_str = ""
        for v, w in zip(row_values, col_widths):
            row_str += str(v).center(w)
        print(row_str)

    print("=" * sum(col_widths))
    print()


def save_results(
    results: List[Dict[str, Any]],
    output_path: str = "results.json",
) -> None:
    """Save results to JSON file."""

    def convert(obj):
        """Make numpy types JSON-serializable."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to {path.resolve()}")
