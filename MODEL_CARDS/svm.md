# Model Card: Classical Support Vector Machine (RBF)

- **Architecture:** Support Vector Classifier with Radial Basis Function kernel (`scikit-learn`)
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) (569 samples, 30 features)
- **Preprocessing:** Stratified 80/20 split, StandardScaler (train-only), PCA (4 components)
- **Hyperparameters:** `C=1.0`, `kernel='rbf'`, `gamma='scale'`, `probability=True`
- **Cross-Validated Metrics (5-Fold x 3 Seeds):**
  - Accuracy: 95.5% (+/- 1.6%)
  - Sensitivity: 93.4% (+/- 2.5%)
  - Specificity: 96.8% (+/- 1.5%)
  - ROC-AUC: 0.985
- **Risk Stratification:** Platt/Sigmoid probability scaling with dynamic threshold tuning support.
- **Limitations:** RBF kernel scales quadratically with dataset size; fixed Euclidean distance metric.

## Empirical Tuning & Optimization Pass (Part B)
- **Tuned Hyperparameters (Nested GridSearchCV):** `C=2.0`, `gamma='scale'`
- **Before vs After Tuning (5-Fold CV):**
  - Accuracy: Baseline `95.79%` -> Tuned `95.26%` ($\Delta = -0.53\%$, $p = 0.207$, not statistically significant)
  - ROC-AUC: Baseline `0.9916` -> Tuned `0.9918` ($\Delta = +0.02\%$, $p = 0.071$, trend toward improvement)
- **Feature Selection vs PCA:**
  - SelectKBest (Mutual Information, 4 features): Accuracy `93.85%` ($\Delta = -1.93\%$, $p = 0.0108$, PCA is statistically superior)
  - Random Forest Top-4 Importances: Accuracy `94.38%` ($\Delta = -1.41\%$, $p = 0.0989$)
- **Class-Weight Balanced (`class_weight='balanced'`):**
  - Sensitivity: `95.52%` (vs baseline `96.92%`, $p = 0.229$)
