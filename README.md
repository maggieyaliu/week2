# Online Shoppers Purchasing Intention - Baseline Model

This project establishes a predictive baseline for consumer conversion behavior using the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). The goal is to evaluate how well a standard linear model can estimate the likelihood of a purchase based on session-level browsing patterns.

## 1. Project Setup

To run the code, you will need **Python 3.8+** and the following libraries installed:
* `pandas`
* `numpy`
* `scikit-learn`

Install the dependencies via your terminal:
`pip install pandas numpy scikit-learn`

## 2. Data Acquisition

The first step is to load the dataset. The system reads the CSV file from the local directory and verifies the dimensions of the data to ensure the file was read correctly.

## 3. Methodology (Preprocessing)

This baseline focuses on real-time behavioral metrics rather than seasonal trends. 
* **Feature Selection:** The `Month` variable is excluded to remove seasonal bias.
* **Numerical Scaling:** Features like `PageValues` and `ExitRates` are standardized using a `StandardScaler`.
* **Categorical Encoding:** Features like `VisitorType` and `TrafficType` are transformed into binary vectors via `OneHotEncoder`.
* **Validation Strategy:** The dataset is split using an 80/20 ratio. A stratified split is used to account for the class imbalance, as only ~15% of sessions result in revenue.

## 4. Execution (Model Training)

The model is built using a **Logistic Regression** algorithm wrapped in a Scikit-Learn Pipeline. This ensures that the preprocessing steps (scaling and encoding) are applied consistently to the test data without data leakage. The model is trained using a maximum of 1,000 iterations to ensure convergence.

## 5. Evaluation (Expected Results)

The model is evaluated using two primary metrics to establish a rigorous benchmark:
* **ROC AUC Score:** Measures the model's ability to distinguish between a buyer and a non-buyer. The expected baseline is **~0.88**.
* **Classification Report:** Provides Precision and Recall. While Precision is generally high, the baseline **Recall** for purchasers is typically low (**~36%**), marking it as the key area for future improvement.
