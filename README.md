# Heart Disease Prediction System

Python | Pandas | NumPy | Scikit-learn | Streamlit

An end-to-end machine learning system that predicts a patient's heart disease
risk in real time. Built to match the project exactly as described:

- Performed **data cleaning, preprocessing, outlier detection and capping** on
  patient health data to improve data quality.
- Applied **feature engineering and categorical encoding**, and trained
  **Logistic Regression** and **Random Forest** models for heart disease
  prediction.
- Evaluated model performance using **Precision, Recall, F1-score, ROC-AUC,
  and confusion matrix**.
- Developed and deployed an interactive **Streamlit application** for
  real-time patient risk prediction.

## Project Structure

```
heart_disease_project/
├── app.py                     # Streamlit real-time prediction app
├── requirements.txt
├── data/
│   └── heart_disease_raw.csv  # Synthetic UCI-style patient dataset (with
│                               # injected missing values & outliers)
├── src/
│   ├── generate_data.py       # Generates the raw dataset
│   ├── preprocessing.py       # Cleaning, outlier capping, feature
│   │                          # engineering, categorical encoding
│   └── train.py               # Trains + evaluates both models
├── models/                    # Saved trained models (.pkl) - created by train.py
└── reports/                   # Confusion matrices, ROC curve, metrics summary
```

## Dataset

The dataset mirrors the classic **UCI Heart Disease** feature set (age, sex,
chest pain type, resting blood pressure, cholesterol, fasting blood sugar,
resting ECG, max heart rate, exercise-induced angina, ST depression, slope,
number of major vessels, thalassemia, and the target label). It is
synthetically generated with realistic distributions and deliberately
includes missing values and outliers so the cleaning pipeline has real work
to do. To use the real UCI dataset instead, simply replace
`data/heart_disease_raw.csv` with the same 14 columns.

## Pipeline (`src/preprocessing.py`)

1. **Data cleaning** — median imputation for numeric columns, mode imputation
   for categorical columns.
2. **Outlier detection & capping** — IQR method (1.5×IQR rule); values are
   winsorized (capped) rather than dropped, preserving sample size.
3. **Feature engineering** — age group, blood pressure category, cholesterol
   category, heart-rate reserve, and an exercise-risk composite flag.
4. **Categorical encoding** — one-hot encoding of nominal features.

## Models & Evaluation (`src/train.py`)

Trains **Logistic Regression** (with feature scaling) and **Random Forest**,
then reports for each:

- Precision, Recall, F1-score, ROC-AUC
- Confusion matrix (saved as PNG in `reports/`)
- Full classification report
- ROC curve comparison plot (`reports/roc_curve_comparison.png`)
- `reports/metrics_summary.json` with all metrics + the outlier report

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (already included, but re-runnable)
python src/generate_data.py

# 3. Train and evaluate both models
python src/train.py

# 4. Launch the real-time prediction app
streamlit run app.py
```

The Streamlit app loads the saved models from `models/`, lets you switch
between Logistic Regression and Random Forest, enter a patient's clinical
details through a form, and instantly returns a risk prediction with
probability.

## Notes

- Random seed fixed at 42 throughout for reproducibility.
- The Random Forest currently edges out Logistic Regression slightly on
  precision and ROC-AUC on this synthetic data; re-run `train.py` after
  swapping in real UCI data to get production-representative numbers.
- This is a portfolio/demo project — not a medical diagnostic tool.
