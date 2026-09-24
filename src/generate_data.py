"""
generate_data.py
-----------------
Generates a synthetic patient health dataset that mirrors the structure of the
well-known UCI Heart Disease dataset (same 13 clinical features + target).
Deliberately injects a few missing values and outliers so the cleaning /
outlier-capping stage of the pipeline has real work to do.

Run:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_STATE = 42
N_SAMPLES = 500

np.random.seed(RANDOM_STATE)


def generate_heart_disease_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    age = np.random.randint(29, 78, n_samples)
    sex = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])          # 0 = female, 1 = male
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.29, 0.07])  # chest pain type
    trestbps = np.random.normal(131, 17, n_samples).round().astype(int)  # resting blood pressure
    chol = np.random.normal(246, 51, n_samples).round().astype(int)      # serum cholesterol
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])            # fasting blood sugar > 120
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.5, 0.02])
    thalach = np.random.normal(149, 23, n_samples).round().astype(int)   # max heart rate achieved
    exang = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])          # exercise induced angina
    oldpeak = np.abs(np.random.normal(1.0, 1.15, n_samples)).round(1)    # ST depression
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.14, 0.47, 0.39])
    ca = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.58, 0.21, 0.13, 0.06, 0.02])
    thal = np.random.choice([0, 1, 2, 3], n_samples, p=[0.02, 0.06, 0.55, 0.37])

    # Build a signal so target correlates realistically with risk factors
    risk_score = (
        0.03 * age
        + 0.9 * sex
        + 0.5 * (cp == 0).astype(int)
        + 0.02 * (trestbps - 130)
        + 0.015 * (chol - 240)
        + 0.6 * exang
        + 0.4 * oldpeak
        + 0.35 * ca
        - 0.02 * (thalach - 150)
        + np.random.normal(0, 1.2, n_samples)
    )
    target = (risk_score > np.median(risk_score)).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target,
    })

    # --- Inject missing values (to be handled by data cleaning) ---
    for col in ["trestbps", "chol", "thalach", "ca", "thal"]:
        missing_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    # --- Inject outliers (to be handled by outlier detection & capping) ---
    outlier_idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[outlier_idx[:4], "chol"] = np.random.randint(560, 650, 4)      # extreme cholesterol
    df.loc[outlier_idx[4:], "trestbps"] = np.random.randint(200, 240, 4)  # extreme blood pressure

    return df


if __name__ == "__main__":
    df = generate_heart_disease_data()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "heart_disease_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
