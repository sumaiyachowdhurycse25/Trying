import streamlit as st
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("Models/clinical_diagnosis_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Streamlit UI
st.title("🩺 Clinical Diagnosis Prediction")
st.write("Enter clinical notes to predict diagnosis.")

clinical_text = st.text_area("Clinical Notes", height=200)

if st.button("Predict Diagnosis"):
    if clinical_text.strip() == "":
        st.warning("Please enter clinical notes before predicting.")
    else:
        prediction = model.predict([clinical_text])[0]
        st.success(f"Predicted Diagnosis: **{prediction}**")

