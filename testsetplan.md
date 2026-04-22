# Locked Test Set Plan

## 1. The "Firewall" Strategy
To prevent data leakage during the AutoResearch loop, the dataset will be split into two distinct environments:
* **The Development Sandbox (80%):** This is the only data the AutoResearch agent is allowed to "see." It will use this for training and experimentation
* **The Vault (20%):** This is the **Locked Test Set**. It will be exported to a separate file (`locked_test_set.csv`) and will **never** be used for training, feature selection, or hyperparameter tuning.

## 2. Technical Implementation
We use a **Stratified Split** with a fixed `random_state` to ensure that the test set is representative of the rare "Purchase" class (~15.4%) and is perfectly reproducible.

```python
from sklearn.model_selection import train_test_split

# Load original data
df = pd.read_csv('online_shoppers_intention.csv')

# Create the Vault (Locked Test Set)
# We use stratify=df['Revenue'] to ensure the purchase rate is identical in both sets
train_dev_df, locked_test_df = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['Revenue']
)

# Export to prevent accidental access
train_dev_df.to_csv('dev_sandbox.csv', index=False)
locked_test_df.to_csv('locked_test_set.csv', index=False)
