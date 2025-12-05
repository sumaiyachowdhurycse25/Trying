import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------------
# Load YOLO model one time
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🏡 Virtual Staging")
st.write("Upload a room image and choose a staging suggestion.")

# -------------------------------------------------------------
# Manual style selector
# -------------------------------------------------------------
style = st.selectbox(
    "Choose staging suggestion",
    ["None", "Modern", "Minimalist", "Scandinavian", "Industrial", "Luxury"]
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------------
# Rule-based suggestion function
# -------------------------------------------------------------
def suggest_style(detected_classes):
    if "bed" in detected_classes:
        return "Minimalist Bedroom"
    if "couch" in detected_classes or "tv" in detected_classes:
        return "Modern Living Room"
    if "chair" in detected_classes and "dining table" in detected_classes:
        return "Scandinavian Dining Room"
    return "Modern Neutral Style"

# -------------------------------------------------------------
# When user uploads image
# -------------------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_np = np.array(image)

    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)

    # Run YOLO detection once
    results = model.predict(image_np, conf=conf)
    annotated_img = results[0].plot()

    st.subheader("Detection Results")
    st.image(annotated_img, caption="Detected Objects", use_container_width=True)

    # ---------------------------------------------------------
    # Extract detected classes
    # ---------------------------------------------------------
    detected_classes = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        detected_classes.append(model.names[cls_id])

    # ---------------------------------------------------------
    # Automatic AI suggestion
    # ---------------------------------------------------------
    ai_suggestion = suggest_style(detected_classes)
    st.success(f"🧠 AI Suggested Staging Style: **{ai_suggestion}**")

    # ---------------------------------------------------------
    # suggestion (if user selected option)
    # ---------------------------------------------------------
    if style != "None":
        st.info(f"🎨User suggested style selection: **{style}**")
