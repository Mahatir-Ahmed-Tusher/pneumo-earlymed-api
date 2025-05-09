import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define class labels
CLASS_NAMES = ["Normal", "Pneumonia"]

# Path to the model
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chest_xray_model.h5")

# Load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model once at startup
model = load_model()

# Preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize to 224x224
    image = image.resize((224, 224))
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Run inference
def run_inference(image: Image.Image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Run prediction
    prediction = model.predict(processed_image)[0][0]  # Single sigmoid output
    # Determine label and confidence
    predicted_class = int(prediction > 0.5)
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    # Prepare the result
    result = {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": confidence * 100  # Percentage
    }
    
    return result