"""
Project 05: Credit Card Fraud Detection
Handles severe class imbalance with SMOTE + Logistic Regression.
Original, reproducible implementation using sklearn's make_classification.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
import joblib
import os


def generate_fraud_data(n_samples: int = 10000, fraud_ratio: float = 0.02, seed: int = 42):
    """Create an imbalanced binary classification dataset simulating fraud."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.01,
        random_state=seed
    )
    feature_names = [f"V{i}" for i in range(1, 21)]
    df = pd.DataFrame(X, columns=feature_names)
    df["Class"] = y  # 1 = fraud
    return df


def train_fraud_model(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
    
    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Fraud Detection Results ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC          : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.joblib")
    print("\nModel saved to models/fraud_model.joblib")
    
    return model


if __name__ == "__main__":
    df = generate_fraud_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean()*100:.2f}%")
    train_fraud_model(df)
