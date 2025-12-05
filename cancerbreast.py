import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('Models/breast_cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Breast Cancer Prediction")
st.write("""
This app predicts whether a tumor is **Malignant** or **Benign** based on user input features.
""")

# Features used in your trained model
with open('Models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Create input widgets dynamically with unique keys
user_input = {}
for i, feature in enumerate(feature_names):
    # Use sliders instead of text input for better UX
    user_input[feature] = st.number_input(
        label=f"{feature}", 
        min_value=0.0, 
        max_value=100.0,  # adjust max if needed
        value=1.0, 
        step=0.01,
        key=f"{feature}_{i}"  # unique key to avoid StreamlitDuplicateElementId
    )

# Convert inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    
    # Map output to human-readable label
    result = "Malignant" if prediction == 1 else "Benign"
    
    st.markdown(f"### Prediction: **{result}**")
