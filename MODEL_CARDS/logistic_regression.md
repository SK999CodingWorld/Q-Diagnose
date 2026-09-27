# Model Card: Logistic Regression Baseline

- **Architecture:** Classical L2-regularized Logistic Regression (`scikit-learn`)
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) (569 samples, 30 features)
- **Preprocessing:** Stratified 80/20 train/test split, StandardScaler (train-only), PCA (4 components)
- **Hyperparameters:** `C=1.0`, `solver='lbfgs'`, `max_iter=1000`
- **Cross-Validated Metrics (5-Fold x 3 Seeds):**
  - Accuracy: 95.8% (+/- 1.4%)
  - Sensitivity: 93.9% (+/- 2.1%)
  - Specificity: 97.0% (+/- 1.2%)
  - ROC-AUC: 0.988
- **Risk Stratification:** Sigmoid probabilities mapped to Low (<30%), Moderate (30-60%), High (60-85%), Very High (>=85%)
- **Limitations:** Linear decision boundary in PCA space; cannot capture higher-order quantum feature interactions.

## Empirical Tuning Pass (Part B)
- **Tuned Hyperparameters (Nested GridSearchCV):** `C=10.0`, `penalty='l2'`
- **Before vs After Tuning (5-Fold CV):**
  - Accuracy: Baseline `96.49%` -> Tuned `96.31%` ($\Delta = -0.18\%$, $p = 0.3739$, not statistically significant)
  - ROC-AUC: Baseline `0.9943` -> Tuned `0.9939` ($p = 0.421$, not statistically significant)
- **Class-Weight Balanced:** Accuracy `95.61%`, Sensitivity `95.80%`
