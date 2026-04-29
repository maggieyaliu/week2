# Experiment Log: Baseline Model Establishment
**Date:** April 22, 2026

## 1. Experiment Objective
To establish a rigorous predictive benchmark using a simplified linear pipeline. The goal is to determine the predictive power of real-time behavioral features (browsing activity) while removing seasonal temporal bias by excluding the `Month` variable.

## 2. Configuration & Methodology
* **Model:** Logistic Regression (L2 Penalty, max_iter=1000)
* **Data Split:** 64% Training / 16% Validation / 20% Validation (Stratified)
* **Features Included:** 10 Numerical (Scaled), 6 Categorical (One-Hot Encoded)
* **Key Exclusion:** `Month` (Removed to isolate behavioral intent from seasonal trends)

## 3. Results (Quantitative)
The model achieved high overall accuracy but revealed a significant struggle with minority class detection.

| Metric | Result |
| :--- | :--- |
| **ROC AUC Score** | **0.8793** |

## 4. Next Steps for AutoResearch Agent
The baseline is now locked at **0.8793 AUC**. The next research cycle will focus on:
1.  **Non-linear Modeling:** Implementing Gradient Boosted Trees (XGBoost/LightGBM) to capture interactions between `ExitRates` and `ProductRelated_Duration`.
2.  **Synthetic Balancing:** Testing SMOTE or class-weight adjustments to address the 34% Recall floor.
3.  **Feature Interaction:** Investigating if specific combinations of `TrafficType` and `PageValues` yield higher signal.



---

# Experiment Log: AutoResearch Iterations
**Date:** April 28, 2026

## 1. Experiment Objective
Run three autoresearch iterations on `model.py` to improve validation ROC AUC while preserving a valid sklearn-compatible pipeline and keeping runtime practical on CPU.

## 2. Configuration & Methodology
* **Scope Constraint:** Modified `model.py` only
* **Evaluation Data/Metric:** Existing frozen split and metrics from `prepare.py` (Validation ROC AUC, Validation F1)
* **Approach:** Tested 3 candidate model configurations and retained the highest-AUC result

## 3. Results (Quantitative)

| Iteration | Model Summary | Validation AUC | Validation F1 |
| :--- | :--- | :---: | :---: |
| **Iter 1** | Logistic Regression (`class_weight='balanced'`) | **0.891852** | 0.640327 |
| **Iter 2** | Random Forest (`n_estimators=350`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`) | **0.915401** | **0.655008** |
| **Iter 3** | HistGradientBoosting + dense one-hot preprocessing | 0.915381 | 0.635548 |

## 4. Outcome
Best configuration from today: **Iteration 2 (Random Forest)** with **AUC = 0.915401**.

This replaces the prior `model.py` configuration because it achieved the highest validation AUC across the three iterations.

## 5. Next Steps
1. Tune Random Forest depth/leaf constraints and `max_features` around the winning setup.
2. Try probability calibration (`CalibratedClassifierCV`) to potentially improve threshold-sensitive performance.
3. Compare with a tuned gradient-boosted tree under the same preprocessing and logging protocol.
