# Model Card: XGBoost (Gradient Boosted Trees)

- **Architecture:** Gradient boosted decision trees (`xgboost`)
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic)
- **Preprocessing:** Zero-leakage StandardScaler and PCA projection (4 components)
- **Hyperparameters:** `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `eval_metric='logloss'`
- **Cross-Validated Metrics:**
  - Accuracy: 95.1% (+/- 1.5%)
  - Sensitivity: 93.0% (+/- 2.2%)
  - Specificity: 96.4% (+/- 1.4%)
  - ROC-AUC: 0.984
- **Risk Stratification:** Logistic margin output mapped to 4-tier risk stratification.
- **Limitations:** Black-box nature requiring SHAP explanations for clinical auditability.

## Empirical Tuning Pass (Part B)
- **Tuned Hyperparameters (Nested GridSearchCV):** `learning_rate=0.2`, `max_depth=2`, `n_estimators=100`
- **Before vs After Tuning (5-Fold CV):**
  - Accuracy: Baseline `93.50%` -> Tuned `93.68%` ($\Delta = +0.18\%$, $p = 0.6213$, not statistically significant)
  - ROC-AUC: Baseline `0.9877` -> Tuned `0.9874` ($\Delta = -0.02\%$, $p = 0.8479$, not statistically significant)
- **Key Insight:** Reducing max tree depth from 4 to 2 regularizes gradient boosting on 4 PCA components, yielding a modest +0.18% gain in accuracy without statistically changing AUC.
