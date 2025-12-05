import streamlit as st

st.set_page_config(
        page_title="Disease Prediction Website",
        page_icon="🏠",
        layout="centered"
    )

st.title("Disease Prediction Website")
st.write("Use the sidebar to navigate through different prediction models.")

st.markdown("---")

st.header("About This Website")
st.write("Select a model from the sidebar to get started.")

    # Example of a page link if you have other pages in a 'pages' directory
st.page_link("pages/strokeapp.py", label="Stroke Prediction", icon="📊")
st.page_link("pages/chatbotapp.py", label="Chatbot", icon="📊")
st.page_link("pages/kidneyapp.py", label="Kidney Diasease Prediction", icon="📊")
st.page_link("pages/thyroidapp.py", label="Thyroid Prediction", icon="📊")
st.page_link("pages/heartapp.py", label="Heart Disease Prediction", icon="❤️")
st.page_link("pages/braintumorapp.py", label="Brain Tumor Prediction", icon="📊")
st.page_link("pages/cancerbreast.py", label="Breast Cancer Prediction", icon="📊")
st.page_link("pages/diabetesapp.py", label="Your Diabetes Risk Prediction", icon="📊")
st.page_link("pages/clinicalnotes.py", label="Write Clinical Notes to Predict Diagnosis", icon="📊")
st.page_link("pages/parkinssonapp.py", label="Find Parkinsson Risk", icon="📊")







