# Model Card: Estimator Quantum Neural Network (QNN)

- **Architecture:** Qiskit `EstimatorQNN` with `NeuralNetworkClassifier`
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic)
- **Qubits / Features:** 4 Qubits, Angle Scaled [0, pi]
- **Circuit Details:** `ZZFeatureMap(reps=1)` + `RealAmplitudes(reps=1)` with Pauli-Z observable
- **Optimizer:** COBYLA (`maxiter=20`)
- **Cross-Validated Metrics:**
  - Accuracy: 71.9% (+/- 3.8%)
  - ROC-AUC: 0.774
- **Risk Stratification:** Sigmoid transformation of continuous expectation values.
- **Limitations:** High sensitivity to parameter initialization; gradient vanishing with increasing qubit count.
