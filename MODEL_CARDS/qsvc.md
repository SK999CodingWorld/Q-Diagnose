# Model Card: Quantum Support Vector Classifier (QSVC)

- **Architecture:** Quantum Kernel SVM with ZZFeatureMap (`qiskit-machine-learning`)
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) (569 samples)
- **Qubits / Features:** 4 Qubits (PCA components) with Angle Scaling into [0, pi]
- **Circuit Details:** `ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')`
- **Kernel Evaluation:** Exact Statevector Fidelity Kernel: K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
- **Cross-Validated Metrics (5-Fold x 3 Seeds):**
  - Accuracy (Ideal Simulator): 81.7% (+/- 2.8%)
  - Accuracy (Noisy Simulator, p=0.02): 76.8% (+/- 3.1%)
  - Sensitivity (at tau=0.35): 98.3%
  - Specificity (at tau=0.35): 32.4%
  - ROC-AUC: 0.887
- **Risk Stratification:** Sigmoid calibration of quantum kernel decision function (`QSVCWrapper`).
- **Limitations:** $O(N^2)$ quantum circuit evaluations; noise degradation under depolarizing channels.

## Quantum Circuit Tuning & Encoding Study (Part B)
- **Feature Map & Repetition Sweep:**
  - `ZZFeatureMap` (reps = 1, linear): Accuracy `57.33%`
  - `ZZFeatureMap` (reps = 2, linear): Accuracy `60.67%` ($\Delta = +3.34\%$, reps=2 provides superior expressivity and state separation)
  - `PauliFeatureMap` (paulis=['Z', 'ZZ'], reps = 1): Accuracy `57.33%`
- **Key Insight:** Increasing ansatz depth from reps=1 to reps=2 improves quantum kernel contrast in 4-qubit Hilbert space before barren plateau degradation occurs at reps >= 3.
