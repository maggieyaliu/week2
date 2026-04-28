"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric, and plotting for 
Online Shoppers Purchasing Intention Project.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import csv
import os

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.2
RESULTS_FILE = "results.tsv"
DATA_PATH = "online_shoppers_WORK.csv"

# ── Data ───────────────────────────────────────────────────
def load_data():
    """Load and split the Shopper Intent WORK dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Ensure you ran the vault split script.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Target: Revenue. Features: Drop target and Month (to avoid seasonal bias)
    X = df.drop(['Revenue', 'Month'], axis=1)
    y = df['Revenue'].astype(int)
    
    # Split into Train and Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=y
    )
    
    return X_train, y_train, X_val, y_val, X.columns.tolist()


# ── Evaluation (frozen metrics) ───────────────────────────
def evaluate(model, X_val, y_val):
    """Compute validation ROC AUC and F1-Score (higher is better)."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    auc = float(roc_auc_score(y_val, y_prob))
    f1 = float(f1_score(y_val, y_pred))
    
    return auc, f1


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_auc, val_f1, status, description):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "val_auc", "val_f1", "status", "description"])
        writer.writerow([experiment_id, f"{val_auc:.6f}", f"{val_f1:.6f}", status, description])


# ── Plotting ───────────────────────────────────────────────
def plot_results(save_path="performance.png"):
    """Plot validation AUC and F1 over experiments."""
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, aucs, f1s, statuses, descriptions = [], [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            aucs.append(float(row["val_auc"]))
            f1s.append(float(row["val_f1"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # ── Top: ROC AUC ──
    ax1.scatter(range(len(aucs)), aucs, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(aucs)), aucs, "k--", alpha=0.2, zorder=2)

    best_auc = []
    current_best_auc = -float("inf")
    for a in aucs:
        current_best_auc = max(current_best_auc, a)
        best_auc.append(current_best_auc)
    ax1.plot(range(len(aucs)), best_auc, color="#2ecc71", linewidth=2.5, label="Best so far")

    ax1.set_ylabel("Validation ROC AUC", fontsize=12)
    ax1.set_title("AutoResearch: Consumer Purchase Behavior", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(min(aucs) * 0.95, 1.0)

    # ── Bottom: F1-Score ──
    ax2.scatter(range(len(f1s)), f1s, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(f1s)), f1s, "k--", alpha=0.2, zorder=2)

    best_f1 = []
    current_best_f1 = -float("inf")
    for f in f1s:
        current_best_f1 = max(current_best_f1, f)
        best_f1.append(current_best_f1)
    ax2.plot(range(len(f1s)), best_f1, color="#2ecc71", linewidth=2.5, label="Best so far")

    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Validation F1-Score", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(f1s) * 0.95, max(f1s) * 1.1)

    # Labels and Legends
    short_labels = [d[:22] + ".." if len(d) > 24 else d for d in descriptions]
    ax2.set_xticks(range(len(aucs)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved tracking plot to {save_path}")


if __name__ == "__main__":
    plot_results()
