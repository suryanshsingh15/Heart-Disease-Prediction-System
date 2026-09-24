Heart Disease Prediction System

A machine learning-based web application for predicting the likelihood of heart disease from patient health attributes. The system combines data preprocessing, machine learning models, model evaluation, and an interactive Streamlit interface.

Overview

This project implements an end-to-end machine learning workflow:

Data → Preprocessing → Model Training → Evaluation → Prediction

The application allows a user to enter patient information through a Streamlit interface and obtain a model-based heart disease prediction.

Disclaimer: This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

Features

* Interactive Streamlit prediction interface
* Data preprocessing and feature scaling
* Logistic Regression model
* Random Forest model
* Saved trained models using Joblib
* Model evaluation using:
    * Precision
    * Recall
    * F1-score
    * ROC-AUC
* Confusion matrix visualizations
* ROC curve comparison
* Outlier detection and handling
* Reproducible Python environment using requirements.txt

Tech Stack

* Python
* Pandas – data manipulation
* NumPy – numerical operations
* Scikit-learn – machine learning and evaluation
* Matplotlib – visualization
* Seaborn – statistical visualization
* Joblib – model serialization
* Streamlit – web application interface

Machine Learning Models

Two classification models are included:

Logistic Regression

A linear classification algorithm used as one of the baseline predictive models.

Random Forest

An ensemble learning algorithm that combines multiple decision trees to perform classification.

Model Evaluation

The models were evaluated using precision, recall, F1-score, and ROC-AUC.

Model	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.7778	0.8400	0.8077	0.8488
Random Forest	0.8478	0.7800	0.8125	0.8528

The repository also contains generated confusion matrices and an ROC curve comparison under the reports/ directory.

Project Structure

Heart-Disease-Prediction-System/
│
├── data/
│   └── heart_disease_raw.csv
│
├── models/
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── confusion_matrix_logistic_regression.*
│   ├── confusion_matrix_random_forest.*
│   ├── metrics_summary.json
│   └── roc_curve_comparison.*
│
├── src/
│   ├── _init_.py
│   ├── generate_data.py
│   ├── preprocessing.py
│   └── train.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

Installation

1. Clone the repository

git clone https://github.com/suryanshsingh15/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System

2. Create a virtual environment

python -m venv .venv

3. Activate the environment

Windows PowerShell:

.venv\Scripts\activate

4. Install dependencies

python -m pip install -r requirements.txt

Running the Application

Start the Streamlit application with:

streamlit run app.py

The application will be available at:

http://localhost:8501

Model Artifacts

The trained models and preprocessing artifacts are stored in the models/ directory.

These include:

* logistic_regression.pkl
* random_forest.pkl
* scaler.pkl
* feature_columns.pkl

The application loads these artifacts to perform predictions without retraining the models every time the application starts.

Evaluation Reports

The reports/ directory contains evaluation outputs generated during model development, including:

* Confusion matrices
* ROC curve comparison
* Model performance metrics
* Outlier processing information

Future Improvements

Potential improvements include:

* Testing additional classification algorithms
* Hyperparameter tuning
* Cross-validation
* Improved user interface and visualization
* Model explainability using techniques such as SHAP
* Deployment using a cloud platform
* Adding automated model evaluation pipelines

Disclaimer

This project is developed for learning and demonstration purposes. Predictions produced by the application should not be interpreted as medical diagnoses. Any health-related decision should be made in consultation with a qualified healthcare professional
