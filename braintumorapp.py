import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. Model Loading Function ---
# Replace this with the actual code to load your specific model
# Ensure the model file is in the same directory or provide the correct path.
@st.cache_resource
def load_my_model():
    # Example for a Keras .h5 model file
    model_path = 'Models/apple4.h5' # Replace with your model file name
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# --- 2. Prediction Function ---
def predict(image):
    if model is None:
        return "Model not available"

    # Preprocess the image to match your model's input requirements
    # Common steps: Resize, convert to numpy array, normalize
    img = image.resize((224, 224)) # Adjust size as per your model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    # Normalize pixel values if your model expects it (e.g., divide by 255.0)
    img_array = img_array / 255.0 

    predictions = model.predict(img_array)
    # Assuming the model outputs a probability for each class
    # The class labels depend on your model's training data (e.g., 'no tumor', 'glioma', etc.)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Define your class labels here
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'] # Adjust as needed

    return class_labels[predicted_class_index]

# --- 3. Streamlit UI ---
st.title("Brain Tumor MRI Classification")
st.header("Upload a brain MRI Image for classification")
st.text("This app classifies brain MRI scans as one of four types: Glioma, Meningioma, Pituitary, or No Tumor.")

uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    
    label = predict(image)
    
    st.success(f"The MRI scan is classified as: **{label}**")
    st.markdown("_Disclaimer: This is not medical advice. Consult a healthcare professional for diagnosis._")


