# Experiment Log: Baseline Model Establishment
**Date:** April 22, 2026

## 1. Experiment Objective
To establish a rigorous predictive benchmark using a simplified linear pipeline. The goal is to determine the predictive power of real-time behavioral features (browsing activity) while removing seasonal temporal bias by excluding the `Month` variable.

## 2. Configuration & Methodology
* **Model:** Logistic Regression (L2 Penalty, max_iter=1000)
* **Data Split:** 80% Training / 20% Testing (Stratified)
* **Features Included:** 10 Numerical (Scaled), 6 Categorical (One-Hot Encoded)
* **Key Exclusion:** `Month` (Removed to isolate behavioral intent from seasonal trends)

## 3. Results (Quantitative)
The model achieved high overall accuracy but revealed a significant struggle with minority class detection.

| Metric | Result |
| :--- | :--- |
| **ROC AUC Score** | **0.8595** |
| **Overall Accuracy** | 88.00% |
| **Precision (Revenue=True)** | 0.75 |
| **Recall (Revenue=True)** | **0.34** |
| **F1-Score (Revenue=True)** | 0.47 |

## 4. Feature Importance (Qualitative)
The top coefficients indicate that session behavior is far more predictive than user demographics:
1.  **PageValues (1.50):** The dominant predictor. Confirms that users visiting historically high-conversion pages are the most likely to purchase.
2.  **ExitRates (-0.76):** A strong negative correlation. As the likelihood of a page being the "last" in a session increases, the probability of revenue drops significantly.
3.  **Technical Metadata:** Interestingly, specific browsers (Browser_12) and traffic types appeared in the top 10, suggesting potential correlations between high-intent marketing channels and conversion.

## 5. Failure Analysis & Findings
* **The "Recall Gap":** While the model is good at ranking (AUC 0.86), it only identifies 34% of actual purchasers. This suggests that a linear decision boundary is too simple to capture the nuanced behaviors of the remaining 66% of buyers.
* **Class Imbalance Impact:** The model is heavily biased toward the majority class (Non-purchasers), as evidenced by the 0.98 recall for `Revenue=0`.

## 6. Next Steps for AutoResearch Agent
The baseline is now locked at **0.8595 AUC**. The next research cycle will focus on:
1.  **Non-linear Modeling:** Implementing Gradient Boosted Trees (XGBoost/LightGBM) to capture interactions between `ExitRates` and `ProductRelated_Duration`.
2.  **Synthetic Balancing:** Testing SMOTE or class-weight adjustments to address the 34% Recall floor.
3.  **Feature Interaction:** Investigating if specific combinations of `TrafficType` and `PageValues` yield higher signal.


