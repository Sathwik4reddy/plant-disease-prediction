import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.9)),
        url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main container (glass effect) */
    .block-container {
        background: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 15px;
    }

    /* Text color */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #ffffff !important;
    }

    /* Upload box styling */
    section[data-testid="stFileUploader"] {
        background: rgba(0,0,0,0.6);
        padding: 15px;
        border-radius: 10px;
    }

    /* Button */
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---- Title ----
st.title("🌿 Plant Disease Prediction")

# ---- Load model ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_prediction_model.h5")

model = load_model()

# ---- Load class labels ----
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# ---- Upload image ----
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Predict"):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        st.success(f"Disease: {class_names[predicted_class]}")
        st.info(f"Confidence: {confidence:.2f}")
