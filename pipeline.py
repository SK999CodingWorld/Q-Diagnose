"""
Q-Diagnose Pipeline Module
===========================
Reusable preprocessing pipeline for the Q-Diagnose platform.
Handles: stratified splitting, scaling (fit on train only),
PCA dimensionality reduction (fit on train only).

No leakage guarantee: scaler and PCA are fit ONLY on training data.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PipelineResult:
    """Container for pipeline outputs."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    pca: PCA
    feature_names: list
    n_components: int
    explained_variance_ratio: np.ndarray


def load_breast_cancer_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """Load the UCI Breast Cancer Wisconsin (Diagnostic) dataset."""
    data = load_breast_cancer()
    X = data.data          # (569, 30)
    y = data.target        # binary: 0=malignant, 1=benign
    feature_names = list(data.feature_names)
    return X, y, feature_names


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_names: Optional[list] = None,
) -> PipelineResult:
    """
    Execute the full preprocessing pipeline.

    Steps (order matters for no-leakage guarantee):
    1. Stratified train/test split
    2. StandardScaler fit on train, transform both
    3. PCA fit on train, transform both

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_components : int, number of PCA components (= number of qubits)
    test_size : float, fraction held out for testing
    random_state : int, reproducibility seed
    feature_names : list of str, optional

    Returns
    -------
    PipelineResult with transformed data and fitted objects
    """
    # Step 1: Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Step 2: Scale — fit on train ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only, no fit

    # Step 3: PCA — fit on train ONLY
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)  # transform only, no fit

    return PipelineResult(
        X_train=X_train_pca,
        X_test=X_test_pca,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        pca=pca,
        feature_names=feature_names or [],
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def prepare_data(
    n_components: int = 4,
    random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:
    """
    Prepare breast cancer dataset with zero-leakage pipeline.
    
    Loads sklearn.datasets.load_breast_cancer(), performs stratified 80/20 split,
    fits StandardScaler on train only, and fits PCA(n_components) on train only.
    
    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, pca
    """
    X, y, _ = load_breast_cancer_data()
    res = run_pipeline(
        X, y,
        n_components=n_components,
        test_size=0.2,
        random_state=random_state
    )
    return res.X_train, res.X_test, res.y_train, res.y_test, res.scaler, res.pca


def run_pipeline_cv_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_components: int = 4,
) -> PipelineResult:
    """
    Execute the pipeline for a single cross-validation fold.

    Fit scaler and PCA ONLY on training indices.

    Parameters
    ----------
    X : full feature array
    y : full label array
    train_idx : indices of training samples for this fold
    test_idx : indices of test samples for this fold
    n_components : PCA components

    Returns
    -------
    PipelineResult
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale — fit on train ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA — fit on train ONLY
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return PipelineResult(
        X_train=X_train_pca,
        X_test=X_test_pca,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        pca=pca,
        feature_names=[],
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
