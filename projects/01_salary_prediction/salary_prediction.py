"""
Project 01: Salary Prediction using Linear Regression
Original implementation for Data Science portfolio.
Uses a synthetic but realistic dataset generator for reproducibility.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def generate_salary_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic salary data with realistic relationships."""
    rng = np.random.default_rng(seed)
    years_experience = rng.uniform(0, 20, n_samples)
    education_level = rng.integers(1, 5, n_samples)  # 1=HighSchool ... 4=PhD
    age = 22 + years_experience + rng.normal(0, 3, n_samples)
    age = np.clip(age, 22, 65)
    
    # Base salary + experience effect + education premium + noise
    salary = (
        30000
        + years_experience * 2500
        + education_level * 8000
        + (age - 30) * 200
        + rng.normal(0, 5000, n_samples)
    )
    salary = np.clip(salary, 25000, 200000)
    
    return pd.DataFrame({
        "years_experience": years_experience.round(1),
        "education_level": education_level,
        "age": age.round(0).astype(int),
        "salary": salary.round(0).astype(int)
    })


def train_and_evaluate(df: pd.DataFrame):
    X = df[["years_experience", "education_level", "age"]]
    y = df["salary"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }
    
    print("=== Salary Prediction - Linear Regression ===")
    print(f"MAE  : ${metrics['MAE']:,.2f}")
    print(f"RMSE : ${metrics['RMSE']:,.2f}")
    print(f"R²   : {metrics['R2']:.4f}")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_.round(2)))}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/salary_model.joblib")
    print("\nModel saved to models/salary_model.joblib")
    
    return model, metrics


if __name__ == "__main__":
    df = generate_salary_data()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print()
    train_and_evaluate(df)
