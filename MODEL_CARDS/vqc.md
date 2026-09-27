# Model Card: Variational Quantum Classifier (VQC)

- **Architecture:** Variational Quantum Circuit with RealAmplitudes and COBYLA optimizer
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic)
- **Qubits / Features:** 4 Qubits, Angle Scaled [0, pi]
- **Circuit Details:** `ZZFeatureMap(reps=1)` + `RealAmplitudes(reps=1, entanglement='linear')`
- **Optimizer:** COBYLA (`maxiter=20`), Subsampled Estimator Wrapper (60 samples)
- **Cross-Validated Metrics:**
  - Accuracy: 72.4% (+/- 3.5%)
  - Sensitivity: 75.0%
  - Specificity: 70.8%
  - ROC-AUC: 0.782
- **Risk Stratification:** Softmax expectation value mapped to risk tiers.
- **Limitations:** Susceptible to Barren Plateaus when depth > 2; non-convex optimization landscape.
