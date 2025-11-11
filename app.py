import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

st.set_page_config(page_title="Hand Fracture Detection", layout="centered")

st.title("🩻 Hand Fracture Detection using YOLOv8")
st.write("Upload a hand X-ray image and the model will detect fractures.")

# Load the YOLO model
@st.cache_resource
def load_model():
    model = YOLO("model.pt")  # make sure best.pt is in the same folder
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    st.write("🔍 Detecting fracture...")

    # Run YOLO prediction
    results = model.predict(source=temp_path, conf=0.25, save=False)

    # Plot result
    result_image = results[0].plot()
    st.image(result_image, caption="Detection Result", use_container_width=True)

    # Show details
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.success(f"✅ Fracture detected! ({len(boxes)} region(s) found)")
    else:
        st.info("❌ No fracture detected in this image.")