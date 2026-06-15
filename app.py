import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="ASTRA Plant Health Detection",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .disease-name {
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌿 ASTRA Plant Health AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a photo of a cotton leaf to instantly detect diseases using Deep Learning.</div>', unsafe_allow_html=True)

# Define Classes for Model 2
MODEL2_CLASSES = {
    0: "Aphids Disease",
    1: "Armyworm Disease",
    2: "Bacterial Blight Disease",
    3: "Powdery Mildew Disease",
    4: "Target Spot Disease"
}

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

@st.cache_resource
def load_disease_model():
    model_path = "model2_final_cotton_disease.h5"
    if not os.path.exists(model_path):
        return None
    return load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)

model = load_disease_model()

if model is None:
    st.error("Model file `model2_final_cotton_disease.h5` not found in the repository. Please make sure it is uploaded.")
    st.stop()

def crop_leaf(img_cv):
    """
    Detects the green leaf in the image and crops it.
    Matches the logic in the ASTRA-Project notebook.
    """
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Green color range (best for cotton leaves)
    lower_green = np.array([25, 40, 20])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean mask
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Largest green object = leaf
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    if w * h < 5000:
        return None

    # Crop with padding
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_cv.shape[1], x + w + pad)
    y2 = min(img_cv.shape[0], y + h + pad)

    crop = img_cv[y1:y2, x1:x2]
    return crop

def preprocess_leaf(img_cv):
    img = cv2.resize(img_cv, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Input Options
option = st.radio("Choose Input Method:", ("Upload Image", "Take a Picture"))

image_data = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = Image.open(uploaded_file)
elif option == "Take a Picture":
    camera_file = st.camera_input("Take a picture of the plant leaf")
    if camera_file is not None:
        image_data = Image.open(camera_file)

if image_data is not None:
    # Display the input image
    st.image(image_data, caption="Input Image", use_container_width=True)
    
    st.write("🔍 Analyzing image...")
    
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Isolating leaf and running neural network..."):
        # Try cropping first
        leaf_img = crop_leaf(img_cv)
        
        if leaf_img is None:
            st.warning("Could not automatically isolate a green leaf. Proceeding with the full image.")
            leaf_img = img_cv
            
        # Preprocess and Predict
        processed_img = preprocess_leaf(leaf_img)
        prediction = model.predict(processed_img, verbose=0)[0]
        
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100
        disease_name = MODEL2_CLASSES[class_idx]
        
        # Display Results
        if confidence > 70:
            color = "#D32F2F" # Red for disease
            icon = "🦠"
        else:
            color = "#F57C00" # Orange for uncertain
            icon = "⚠️"
            
        st.markdown(f"""
        <div class="prediction-box" style="background-color: #FFF3E0; border-left: 5px solid {color};">
            <h3 style="color: {color}; margin: 0;">{icon} Detection Result</h3>
            <div class="disease-name" style="color: {color};">{disease_name}</div>
            <p style="margin-top: 10px; font-size: 16px; color: #555;">Confidence: <b>{confidence:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(int(confidence))
        
        # Show cropped leaf for debugging/transparency
        st.write("---")
        with st.expander("View Preprocessed Leaf Image"):
            st.image(cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB), caption="Cropped Leaf passed to Model")
