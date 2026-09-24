"""
preprocessing.py
-----------------
Implements the full preprocessing pipeline described in the project summary:

    1. Data cleaning (missing-value imputation)
    2. Outlier detection & capping (IQR method)
    3. Feature engineering
    4. Categorical encoding

Every function returns a fresh copy of the DataFrame so the pipeline stays
side-effect-free and easy to test step by step.
"""

import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# --------------------------------------------------------------------------- #
# 1. Data cleaning
# --------------------------------------------------------------------------- #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric columns, mode for categorical."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


# --------------------------------------------------------------------------- #
# 2. Outlier detection & capping (IQR method)
# --------------------------------------------------------------------------- #
def detect_and_cap_outliers(df: pd.DataFrame, cols=None, factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using the IQR rule and cap them (winsorize) to the
    [Q1 - factor*IQR, Q3 + factor*IQR] bounds instead of dropping rows,
    preserving sample size while controlling extreme values.
    """
    df = df.copy()
    cols = cols or NUMERIC_COLS
    report = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
        report[col] = {"lower_bound": round(lower, 2), "upper_bound": round(upper, 2), "outliers_capped": n_outliers}
    df.attrs["outlier_report"] = report
    return df


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive clinically meaningful features from the raw measurements."""
    df = df.copy()

    # Age bracket (captures non-linear age risk)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 40, 50, 60, 120], labels=["<40", "40-50", "50-60", "60+"]
    )

    # Blood pressure category
    df["bp_category"] = pd.cut(
        df["trestbps"], bins=[0, 120, 140, 300], labels=["normal", "elevated", "high"]
    )

    # Cholesterol category
    df["chol_category"] = pd.cut(
        df["chol"], bins=[0, 200, 240, 700], labels=["desirable", "borderline", "high"]
    )

    # Max heart rate reserve relative to age-predicted max (220 - age)
    df["hr_reserve"] = (220 - df["age"]) - df["thalach"]

    # Composite risk flag: exercise-induced angina + high ST depression
    df["exercise_risk_flag"] = ((df["exang"] == 1) & (df["oldpeak"] > 1.0)).astype(int)

    return df


# --------------------------------------------------------------------------- #
# 4. Categorical encoding
# --------------------------------------------------------------------------- #
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categorical/engineered columns."""
    df = df.copy()
    cols_to_encode = ["cp", "restecg", "slope", "thal", "age_group", "bp_category", "chol_category"]
    cols_to_encode = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def preprocess_pipeline(df: pd.DataFrame):
    """Run the full cleaning -> outlier capping -> feature engineering -> encoding pipeline."""
    df_clean = handle_missing_values(df)
    df_capped = detect_and_cap_outliers(df_clean)
    df_feat = engineer_features(df_capped)
    df_encoded = encode_categorical(df_feat)
    return df_encoded, df_capped.attrs.get("outlier_report", {})
