import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("Models/parkinsons_model.pkl", "rb"))

st.title("🧠 Parkinson’s Disease Prediction App")
st.write("Enter to predict Parkinson’s disease")

# Correct 12 feature names (matching model)
feature_names = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Shimmer",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2"
]

st.header("🔧 Input Features")
inputs = []

# Streamlit inputs for all 12 features
for feature in feature_names:
    value = st.number_input(feature, value=0.0, format="%.5f")
    inputs.append(value)

# Convert to proper shape
sample = np.array(inputs).reshape(1, -1)

# Prediction button
if st.button("Predict Parkinson's"):
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    if prediction == 1:
        st.error(f"🧪 Parkinson’s Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ No Parkinson’s Detected (Probability of Parkinson’s: {prob:.2f})")
