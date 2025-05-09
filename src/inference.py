import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define class labels
CLASS_NAMES = ["Normal", "Pneumonia"]

# Path to the model
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chest_xray_model.h5")

# Load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    logger.info("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
    return model

# Load the model once at startup
model = load_model()

# Preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    logger.info("Preprocessing image...")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    logger.info(f"Preprocessed image shape: {image_array.shape}")
    return image_array

# Run inference
def run_inference(image: Image.Image):
    logger.info("Running inference...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Single sigmoid output
    predicted_class = int(prediction > 0.5)
    confidence = prediction if prediction > 0.5 else 1 - prediction
    result = {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": confidence * 100  # Percentage
    }
    logger.info(f"Inference result: {result}")
    return result
