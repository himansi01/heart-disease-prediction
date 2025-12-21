# heart-disease-prediction

Overview

This project is an academic B.Tech final-year project focused on predicting heart disease risk using machine learning and deep learning techniques. It demonstrates end-to-end ML workflow, from data preprocessing to model evaluation and deployment, while also integrating explainable AI for interpretability.


Dataset

Source: UCI Machine Learning Repository – Heart Disease Dataset
Size: 303 patient records
Features: 14 clinical and demographic attributes (e.g., age, sex, cholesterol, blood pressure, max heart rate)
Target: 1 = heart disease, 0 = no heart disease


Models Implemented

Logistic Regression – linear baseline for classification
Random Forest – ensemble model reducing variance
Gradient Boosting – ensemble model with sequential error correction (best generalization)
Artificial Neural Network (ANN) – captures non-linear feature relationships


Model Evaluation

All models were evaluated using multiple metrics for robust assessment:
Accuracy
Precision & Recall
F1-score
ROC–AUC
Confusion Matrix
Key Result: Gradient Boosting achieved high ROC–AUC (~0.99) with stable predictions.


Explainability

Used SHAP (Shapley Additive Explanations) to interpret model predictions.
Highlights important features contributing to heart disease risk, making the model transparent and clinically interpretable.


Deployment

Developed a Streamlit web application for real-time predictions.
Users can input clinical parameters and get an instant risk prediction:
"You are at high risk for heart disease." OR "You are safe."


Technologies & Libraries

Programming Language: Python
Data & Analysis: Pandas, NumPy, Matplotlib, Seaborn
Machine Learning: Scikit-learn
Deep Learning: TensorFlow / Keras
Explainability: SHAP
Deployment: Streamlit




