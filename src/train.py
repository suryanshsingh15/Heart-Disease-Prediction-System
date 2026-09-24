"""
train.py
--------
Trains Logistic Regression and Random Forest models on the preprocessed
heart disease data, evaluates each with Precision, Recall, F1-score,
ROC-AUC, and a confusion matrix, and saves the fitted models + scaler
for use by the Streamlit app.

Run:
    python src/train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

from preprocessing import preprocess_pipeline

RANDOM_STATE = 42
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease_raw.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"Precision : {metrics['precision']}")
    print(f"Recall    : {metrics['recall']}")
    print(f"F1-score  : {metrics['f1_score']}")
    print(f"ROC-AUC   : {metrics['roc_auc']}")
    print("Confusion Matrix:")
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # Confusion matrix plot
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"), dpi=150)
    plt.close()

    return metrics, y_proba


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load raw data
    df_raw = pd.read_csv(DATA_PATH)

    # Full preprocessing pipeline: cleaning -> outlier capping -> feature engineering -> encoding
    df_processed, outlier_report = preprocess_pipeline(df_raw)
    print("Outlier capping report (IQR method):")
    print(json.dumps(outlier_report, indent=2))

    X = df_processed.drop(columns=["target"])
    y = df_processed["target"]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale numeric features (helps Logistic Regression converge well)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Logistic Regression ----
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    lr_metrics, lr_proba = evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test)

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)  # tree-based model doesn't need scaling
    rf_metrics, rf_proba = evaluate_model("Random Forest", rf, X_test, y_test)

    # ROC curve comparison plot
    plt.figure(figsize=(5.5, 4.5))
    for name, y_proba in [("Logistic Regression", lr_proba), ("Random Forest", rf_proba)]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve_comparison.png"), dpi=150)
    plt.close()

    # Save models, scaler, and feature schema for the Streamlit app
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    # Save metrics summary
    summary = {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "outlier_report": outlier_report,
    }
    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nModels, scaler, and reports saved successfully.")
    print(f"  -> {MODELS_DIR}")
    print(f"  -> {REPORTS_DIR}")


if __name__ == "__main__":
    main()
