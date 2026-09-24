"""
app.py
------
Streamlit application for real-time heart disease risk prediction.
Loads the trained Logistic Regression / Random Forest models and lets the
user pick which one to use, enter patient details through a form, and get
an instant prediction with probability.

Run:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import handle_missing_values, detect_and_cap_outliers, engineer_features

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="centered")


@st.cache_resource
def load_artifacts():
    log_reg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return log_reg, rf, scaler, feature_columns


def build_input_row(inputs: dict) -> pd.DataFrame:
    """Turn the raw form inputs into a single-row DataFrame matching the raw schema."""
    return pd.DataFrame([inputs])


def prepare_for_model(raw_row: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Run the same cleaning -> outlier capping -> feature engineering -> encoding steps used in training."""
    df = handle_missing_values(raw_row)
    df = detect_and_cap_outliers(df)
    df = engineer_features(df)
    df = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal", "age_group", "bp_category", "chol_category"])

    # Align columns with the training-time schema (missing dummy cols = 0, drop extras)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df


def main():
    st.title("❤️ Heart Disease Risk Prediction System")
    st.caption(
        "Data cleaning • outlier capping • feature engineering • categorical encoding • "
        "Logistic Regression & Random Forest"
    )

    log_reg, rf, scaler, feature_columns = load_artifacts()

    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])
    st.sidebar.markdown("---")
    st.sidebar.write("Trained on a UCI-style 13-feature clinical dataset.")

    st.subheader("Patient Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 54)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type", [0, 1, 2, 3],
            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 650, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox(
            "Resting ECG Results", [0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
        )

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment", [0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
        )
        ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia", [0, 1, 2, 3],
            format_func=lambda x: ["Unknown", "Fixed Defect", "Normal", "Reversible Defect"][x],
        )

    if st.button("Predict Risk", type="primary", use_container_width=True):
        raw_inputs = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg,
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        raw_row = build_input_row(raw_inputs)
        X_row = prepare_for_model(raw_row, feature_columns)

        if model_choice == "Logistic Regression":
            X_row_scaled = scaler.transform(X_row)
            proba = log_reg.predict_proba(X_row_scaled)[0, 1]
            pred = log_reg.predict(X_row_scaled)[0]
        else:
            proba = rf.predict_proba(X_row)[0, 1]
            pred = rf.predict(X_row)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ Elevated Risk of Heart Disease  —  probability: {proba:.1%}")
        else:
            st.success(f"✅ Low Risk of Heart Disease  —  probability: {proba:.1%}")
        st.progress(min(float(proba), 1.0))
        st.caption(f"Prediction generated using **{model_choice}**. This is a demo tool, not medical advice.")


if __name__ == "__main__":
    main()
