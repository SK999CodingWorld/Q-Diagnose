# Q-Diagnose: System Architecture & Data Flow

> **Smart India Hackathon 2025 | Problem Statement ID: 26139**  
> **Organization:** Egreen Quanta | **Theme:** MedTech / BioTech / HealthTech  

---

## 1. System Architecture Overview

Q-Diagnose is structured into a modular, decoupled hybrid quantum-classical architecture that separates deterministic classical feature preprocessing from quantum state preparation and Hilbert space projection.

```mermaid
flowchart TD
    subgraph INGESTION["1. Data Ingestion & Governance Layer"]
        CSV[Clinical CSV Ingestion] --> VAL[Schema & Binary Target Validation]
        VAL --> DB[(SQLite Database Metadata Store)]
    end

    subgraph PREPROCESSING["2. Zero-Leakage Classical Preprocessing"]
        VAL --> SPLIT[Stratified 5-Fold Partitioning x 3 Seeds]
        SPLIT --> TRAIN_SPLIT[Training Fold (80%)]
        SPLIT --> TEST_SPLIT[Held-Out Test Fold (20%)]
        TRAIN_SPLIT --> SCALER[Fit StandardScaler]
        SCALER --> PCA_FIT[Fit PCA: N Components = Qubits]
        TEST_SPLIT -.->|Transform Only| SCALER
        TEST_SPLIT -.->|Transform Only| PCA_FIT
        PCA_FIT --> ANGLE[Scale Angles to [0, π]]
    end

    subgraph QUANTUM_LAYER["3. Quantum Processing Unit (QPU) / Simulator"]
        ANGLE --> FM[ZZFeatureMap Encoding (2nd Order Entanglement)]
        FM --> KERNEL[Statevector Quantum Kernel Evaluator]
        FM --> ANSATZ[RealAmplitudes Parameterized Circuit (Depth 1-4)]
        ANSATZ --> COBYLA[COBYLA Classical Optimizer Loop]
        KERNEL --> QSVC[Quantum Support Vector Classifier]
        COBYLA --> VQC[Variational Quantum Classifier]
        COBYLA --> QNN[Estimator Quantum Neural Network]
    end

    subgraph DECISION_SUPPORT["4. Decision Support & Stratification (Gap 1 & 2)"]
        QSVC --> PROB[Sigmoid Decision-Function Calibration]
        VQC --> PROB
        QNN --> PROB
        PROB --> THRESH[Dynamic Operating Threshold (0.01 - 0.99)]
        THRESH --> STRAT[Auditable Risk Stratification: Low, Moderate, High, Very High]
        PROB --> ROC_SWEEP[Dense ROC / PR Curve Operating Sweep]
    end

    subgraph SERVING["5. Presentation & API Layer"]
        DB --> REST[FastAPI REST Backend (Uvicorn 8000)]
        DECISION_SUPPORT --> REST
        REST --> UI[React 18 + Tailwind CSS + Recharts Dashboard]
    end
```

---

## 2. Classical Front-End vs QPU/Simulator Boundary

1. **Classical Responsibilities:**
   - Missing value imputation (median replacement) and duplicate row purging.
   - Out-of-fold `StandardScaler` and `PCA` fitting to prevent dimensional and target leakage.
   - Platt/Sigmoid probability scaling: $P(y=1|x) = \frac{1}{1 + e^{-f(x)}}$.
   - ROC-AUC and PR-AUC metric computation, threshold sweeping, and risk tier categorization.
   - Classical explainability via SHAP (TreeExplainer and KernelExplainer).

2. **Quantum Processing Unit (QPU) / Simulator Responsibilities:**
   - Non-linear feature mapping into $2^N$-dimensional Hilbert space using $N$-qubit parameterized quantum circuits.
   - Quantum kernel matrix generation: $K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$.
   - Quantum variational optimization using `StatevectorSampler` and `StatevectorEstimator` with `COBYLA`.
   - Depolarizing noise simulation via Qiskit Aer `GenericBackendV2` to evaluate NISQ viability.

---

## 3. Data Encoding & Quantum Circuits

- **Feature Map:** `ZZFeatureMap(feature_dimension=N, reps=2, entanglement='linear')`
  - $U_{\Phi(x)} = \exp\left(i \sum_{j} x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$
  - Standard angle encoding scales PCA inputs into $[0, \pi]$ to prevent periodicity aliasing.
- **Ansatz:** `RealAmplitudes(num_qubits=N, reps=d, entanglement='linear')`
  - Alternating layers of parameterized $R_y(\theta)$ rotation gates and linear CNOT entanglement.
