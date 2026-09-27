# Q-Diagnose: Scientific & Operational Limitations

> **Smart India Hackathon 2025 | Problem Statement ID: 26139**  
> **Mandatory Research Disclaimer:** Research prototype only. Not certified for medical diagnosis or clinical patient triage.

---

## 1. Prototype Status & Regulatory Compliance
- **Not a Medical Device:** Q-Diagnose has **not** received FDA 510(k), De Novo, or CE-IVD Software as a Medical Device (SaMD) clearance.
- **Research Use Only:** All probabilistic scores and risk tiers are intended solely for empirical research and clinical decision-support evaluation, never for direct patient management.

---

## 2. Data Limitations
- **Sample Size Constraints:** Benchmarking is currently grounded in the UCI Wisconsin Diagnostic Breast Cancer (WDBC) dataset ($n = 569$) and auxiliary tabular cohorts. While statistically rigorous for benchmarking, real-world clinical deployment requires multi-center cohorts with diverse demographic representations.
- **Dimensionality Reduction Trade-off:** Dimensionality reduction via PCA (reducing $30$ features to $4-8$ qubits) captures $>70\%$ of variance but inherently discards fine-grained non-linear residual signals.

---

## 3. NISQ Hardware & Quantum Simulator Constraints
- **Simulation vs Physical QPU:** Current runs execute on exact statevector simulators and Qiskit Aer noisy backends. Physical NISQ QPUs suffer from thermal relaxation ($T_1$), dephasing ($T_2$), and two-qubit gate errors ($> 10^{-2}$).
- **Barren Plateau Vulnerability:** As shown in our circuit depth sweeps, increasing ansatz repetitions beyond $d=2$ on unconstrained Haar-random variational circuits induces exponential concentration of gradients (Barren Plateaus), degrading trainability.
- **Classical Superiority in Tabular Domains:** Classical ensemble methods (XGBoost, SVM) consistently achieve higher accuracy ($95-97\%$) and lower computational latency (milliseconds vs seconds) than current NISQ algorithms ($77-82\%$).

---

## 4. Privacy, Data Governance & Patient De-identification Notice
- **De-identified Data Standard:** The Patient Details Module (`/patients`, `/patients/{id}/records`, `/patients/{id}/trend`) is designed strictly for research, algorithmic benchmarking, and educational demonstration. All stored patient identifiers default to synthetic UUIDs or anonymized research labels (e.g., `Subject-SIH-9921`).
- **No Protected Health Information (PHI):** The system must **never** be used to ingest unencrypted or un-anonymized Protected Health Information (PHI) or Personally Identifiable Information (PII) subject to HIPAA, GDPR, or India's Digital Personal Data Protection Act (DPDPA 2023) without formal institutional review board (IRB) approval, hospital EHR integration clearance, and zero-knowledge database encryption.
- **Auditability & Traceability:** Longitudinal records store strictly numeric biomedical feature vectors and model predictions to enable auditable research reproducibility without exposing identifying demographic trails.
