import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load Pickled Model & Scaler
# -------------------------------
with open("Models/thyroid_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Scaler/thyroid_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🩺 Thyroid Disease Classification App")
st.write("This app predicts if a patient has a thyroid condition")

# -------------------------------
# User Input Form
# -------------------------------
st.header("Enter Patient Details")

# ⚠️ Replace these input fields with the exact features used in your model.
#     (They MUST match the columns of X after preprocessing)
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ["M", "F"])  # Example categorical
goitre = st.selectbox("Goitre", [0, 1])

# Example lab values
t3 = st.number_input("T3 (ng/dL)")
tt4 = st.number_input("TT4 (µg/dL)")
t4u = st.number_input("T4U")
fti = st.number_input("FTI")

# Convert to DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Goitre": goitre,
    "T3": t3,
    "TT4": tt4,
    "T4U": t4u,
    "FTI": fti
}])

# Process categorical
input_data = pd.get_dummies(input_data, drop_first=True)

# Reindex to match model training columns
# Try loading model feature names
if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
elif hasattr(model, "feature_names"):
    feature_names = model.feature_names
elif hasattr(scaler, "feature_names_in_"):
    feature_names = scaler.feature_names_in_
else:
    st.error("Feature names not found — please add them during training.")
    st.stop()

input_data = input_data.reindex(columns=feature_names, fill_value=0)


# Predict Button
if st.button("Predict"):
    # Scale
    scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.error("⚠️ Patient likely has a thyroid disorder.")
    else:
        st.success("✅ Patient is likely normal (no thyroid disorder).")


