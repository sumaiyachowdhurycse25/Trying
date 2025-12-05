import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
with open("Models/stroke_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 Stroke Prediction App")
st.write("Enter patient information to predict stroke risk.")

# -------------------------------
# User Input Form
# -------------------------------
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", 0, 100, 45)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", 
                         ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
smoking_status = st.selectbox("Smoking Status", 
                              ["formerly smoked", "never smoked", "smokes", "Unknown"])

# -------------------------------
# Map categorical inputs to numbers
# (ensure mapping matches your training code)
# -------------------------------
label_map = {
    "Male": 1, "Female": 0, "Other": 2,
    "No": 0, "Yes": 1,
    "Urban": 1, "Rural": 0,
    "Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 1, "Never_worked": 4,
    "formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0
}

data = pd.DataFrame([{
    "gender": label_map[gender],
    "age": age,
    "hypertension": label_map[hypertension],
    "heart_disease": label_map[heart_disease],
    "ever_married": label_map[ever_married],
    "work_type": label_map[work_type],
    "Residence_type": label_map[residence_type],
    "avg_glucose_level": glucose,
    "bmi": bmi,
    "smoking_status": label_map[smoking_status]
}])

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Stroke"):
    probability = model.predict_proba(data)[0][1] * 100
    st.info(f"🧠 Stroke Risk Probability: {probability:.2f}%")