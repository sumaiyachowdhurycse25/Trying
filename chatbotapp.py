import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
@st.cache_resource
def load_model():
    with open("Models/health_chatbot.pkl", "rb") as f:
        model, vectorizer, df = pickle.load(f)
    return model, vectorizer, df

model, vectorizer, df = load_model()

# ---------------------------
# Chatbot Function
# ---------------------------
def ask_bot(user_input):
    vec = vectorizer.transform([user_input])
    dist, idx = model.kneighbors(vec, n_neighbors=1)
    return df.iloc[idx[0][0]]['Doctor']


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Health Chatbot", layout="centered")

st.title("🩺 Medical Chatbot (AI-Powered)")
st.write("Ask any health-related question and get a response.")

user_input = st.text_input("🗣️ Your Question:")

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = ask_bot(user_input)
        st.success("💬 Doctor's Response:")
        st.write(response)

st.write("---")
st.caption("⚠️ This is not medical advice. Always consult a real doctor.")

