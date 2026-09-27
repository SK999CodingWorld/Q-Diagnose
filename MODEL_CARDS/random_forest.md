# Model Card: Random Forest Classifier

- **Architecture:** Bagged ensemble of 100 decision trees (`scikit-learn`)
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic)
- **Preprocessing:** Zero-leakage StandardScaler and PCA projection (4 components)
- **Hyperparameters:** `n_estimators=100`, `max_depth=None`, `random_state=42`
- **Cross-Validated Metrics:**
  - Accuracy: 94.2% (+/- 1.8%)
  - Sensitivity: 92.1% (+/- 2.3%)
  - Specificity: 95.5% (+/- 1.9%)
  - ROC-AUC: 0.981
- **Risk Stratification:** Tree voting fraction mapped to risk tiers.
- **Limitations:** Stepwise decision surfaces; susceptibility to overfitting small sample sizes.

## Empirical Tuning Pass (Part B)
- **Tuned Hyperparameters (Nested GridSearchCV):** `n_estimators=150`, `max_depth=None`, `min_samples_split=2`
- **Before vs After Tuning (5-Fold CV):**
  - Accuracy: Baseline `94.56%` -> Tuned `94.38%` ($\Delta = -0.17\%$, $p = 0.7071$, not statistically significant)
  - F1-Score: Baseline `0.9563` -> Tuned `0.9552` ($\Delta = -0.11\%$, $p = 0.7468$, not statistically significant)
- **Key Insight:** Standard 100-tree default is already near saturation on 4-dimensional PCA space; deeper/larger ensembles yield marginal change.
