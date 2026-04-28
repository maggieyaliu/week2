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


