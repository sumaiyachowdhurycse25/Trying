import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Step 1: Load saved model, scaler, and feature columns
# -----------------------------
with open('Models/kidney_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Scaler/kidney_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# -----------------------------
# Step 2: Load dataset to get feature types
# -----------------------------
df = pd.read_csv('CSV/kidney_disease.csv')
df.columns = df.columns.str.strip()

target_col = 'classification'  # Replace if different
X_features = df.drop(columns=[target_col], errors='ignore')
if 'id' in X_features.columns:
    X_features = X_features.drop('id', axis=1)

# -----------------------------
# Step 3: Streamlit layout
# -----------------------------
st.title("Kidney Disease Prediction App")
st.write("Enter patient details")

# -----------------------------
# Step 4: Collect user input
# -----------------------------
input_data = {}
for col in X_features.columns:
    if X_features[col].dtype in ['int64', 'float64']:
        value = st.number_input(f"{col}", value=float(X_features[col].mean()))
    else:
        options = X_features[col].dropna().unique().tolist()
        value = st.selectbox(f"{col}", options)
    input_data[col] = value

# -----------------------------
# Step 5: Predict on button click
# -----------------------------
if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Handle categorical variables
    input_df = pd.get_dummies(input_df)

    # Align input columns with training feature columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"Prediction: CKD (High risk)\nProbability: {probability:.2f}")
    else:
        st.success(f"Prediction: No CKD (Low risk)\nProbability: {probability:.2f}")
