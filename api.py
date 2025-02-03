from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("model_best_checkpoint.keras")

# Define class labels
classes = ["glioma", "meningioma", "notumor", "pituitary"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image.reshape(1, 128, 128, 1) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    return {"prediction": predicted_class, "probabilities": prediction.tolist()}

