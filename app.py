import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Hand Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border-color: #FF6B6B;
    }
    .upload-text {
        text-align: center;
        color: #666;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
        padding-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None

# Title and description
st.title("🦴 Hand Fracture Detection System")
st.markdown("""
    <p style='text-align: center; color: #666; font-size: 1.1rem;'>
    Upload an X-ray image to detect and locate hand fractures using AI
    </p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.subheader("ℹ️ About")
    st.info("""
    This application uses a trained YOLO model to detect fractures in hand X-ray images.
    
    **How to use:**
    1. Adjust detection thresholds if needed
    2. Upload an X-ray image
    3. View detection results
    """)

# Load model automatically from repository
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Try to load model from repository
if not st.session_state.model_loaded:
    model_path = "best.pt"  # Model should be in your repository
    if os.path.exists(model_path):
        with st.spinner("Loading model..."):
            st.session_state.model = load_model(model_path)
            if st.session_state.model is not None:
                st.session_state.model_loaded = True
    else:
        st.error("⚠️ Model file 'best.pt' not found in repository. Please ensure best.pt is uploaded to your GitHub repository.")

# Main content area
if not st.session_state.model_loaded:
    st.warning("⚠️ Please upload your model file (best.pt) in the sidebar to begin.")
    st.info("💡 Alternatively, place your best.pt file in the same directory as this script.")
else:
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a hand X-ray image for fracture detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original X-ray Image", use_container_width=True)
            
            # Add detection button
            if st.button("🔍 Detect Fractures", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Convert PIL image to numpy array
                    img_array = np.array(image)
                    
                    # Run prediction
                    results = st.session_state.model.predict(
                        img_array,
                        verbose=False
                    )
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.detection_done = True
    
    with col2:
        st.subheader("📊 Detection Results")
        
        if 'detection_done' in st.session_state and st.session_state.detection_done:
            results = st.session_state.results
            
            # Get annotated image
            annotated_img = results[0].plot()
            # Convert BGR to RGB using numpy (no cv2 needed!)
            annotated_img = annotated_img[:, :, ::-1]
            
            # Display annotated image
            st.image(annotated_img, caption="Detected Fractures", use_container_width=True)
            
            # Display detection statistics
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            
            # Number of detections
            num_detections = len(results[0].boxes)
            
            if num_detections > 0:
                st.success(f"✅ **{num_detections} fracture(s) detected**")
                
                # Create a dataframe with detection details
                st.subheader("Detection Details")
                
                for idx, box in enumerate(results[0].boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = results[0].names[cls]
                    
                    # Display each detection
                    with st.expander(f"Detection #{idx + 1}: {class_name}", expanded=True):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Confidence", f"{conf:.2%}")
                        with col_b:
                            st.metric("Class", class_name)
                        
                        # Bounding box coordinates
                        bbox = box.xyxy[0].cpu().numpy()
                        st.caption(f"Location: [{int(bbox[0])}, {int(bbox[1])}] to [{int(bbox[2])}, {int(bbox[3])}]")
            else:
                st.info("ℹ️ **No fractures detected**")
                st.write("The model did not detect any fractures with the current confidence threshold.")
                st.write("You may try:")
                st.write("- Lowering the confidence threshold in the sidebar")
                st.write("- Using a different image")
                st.write("- Ensuring the image quality is adequate")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button for annotated image
            st.markdown("---")
            
            # Convert to PIL Image for download
            annotated_pil = Image.fromarray(annotated_img)
            
            # Save to bytes
            import io
            buf = io.BytesIO()
            annotated_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Annotated Image",
                data=byte_im,
                file_name="fracture_detection_result.png",
                mime="image/png"
            )
        else:
            st.info("👈 Upload an image and click 'Detect Fractures' to see results")

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #666; font-size: 0.9rem;'>
    Hand Fracture Detection System | Powered by YOLOv8 & Streamlit
    </p>
""", unsafe_allow_html=True)



