import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------------
# Load Scaler and Model
# --------------------------------------------------------
with open("Scaler/diabetes_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Models/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------------
# App Title
# --------------------------------------------------------
st.title("🔍 Diabetes Prediction App")
st.write("Enter patient details below to predict diabetes using Logistic Regression.")

# --------------------------------------------------------
# User Inputs
# --------------------------------------------------------
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)

insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# --------------------------------------------------------
# Prediction
# --------------------------------------------------------
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🔴 High Risk of Diabetes\n**Probability: {probability:.2f}**")
    else:
        st.success(f"🟢 Low Risk of Diabetes\n**Probability: {probability:.2f}**")
