import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_best_checkpoint.keras")

model = load_model()

# Define class labels
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Streamlit UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to classify the tumor type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = np.array(image)
    image = cv2.resize(image, (128, 128))  # Resize
    image = image.reshape(1, 128, 128, 1) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    # Display results
    st.image(uploaded_file, caption="Uploaded MRI Scan", use_container_width=True)
    st.success(f"Prediction: **{predicted_class}**")
    st.write("Class Probabilities:", {cls: f"{prob:.2%}" for cls, prob in zip(classes, prediction[0])})

