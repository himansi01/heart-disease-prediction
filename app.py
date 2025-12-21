import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")
st.write("This AI model predicts whether a person is at risk of heart disease based on health parameters.")

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.sidebar.header("Enter Your Health Details")


def user_input():
    age = st.sidebar.number_input("Age", 20, 100, 45)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [1, 0])
    restecg = st.sidebar.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate", 50, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1/0)", [1, 0])
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thal (0–3)", [0, 1, 2, 3])

    data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                        columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                                 "oldpeak", "slope", "ca", "thal"])
    return data


df = user_input()

if st.button("Predict"):
    scaled = scaler.transform(df)
    result = model.predict(scaled)[0]
    if result == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ No Heart Disease Risk Detected")
