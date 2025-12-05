import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Model and Scaler
# ----------------------------
with open("Models/heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Scaler/heart_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease.")

# ----------------------------
# Input Fields
# (restecg was dropped)
# ----------------------------
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (1 = male, 0 = female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True)", [0, 1])
exang = st.selectbox("Exercise Induced Angina (1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2])

# Collect input into a list
input_data = np.array([[age, sex, cp, trestbps, chol, thalach, fbs,
                        exang, oldpeak, slope, ca, thal]])

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ No Heart Disease Detected")


