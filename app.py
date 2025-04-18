from gevent import monkey
monkey.patch_all()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image

app = Flask(__name__)

# Load the trained CNN model
model = tf.keras.models.load_model("model/cnn_model.keras")

def preprocess_image(image_data):
    """Convert image to grayscale, resize, normalize, and reshape for model input."""
    image = Image.open(io.BytesIO(image_data)).convert("L")  # Convert to grayscale
    image = image.resize((480, 480))  # Resize to 480x480
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Reshape to (1, 480, 480, 1)

    return image_array

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def home():
    """Serve the frontend HTML."""
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return CNN model prediction."""
    try:
        print("📢 Received request at /predict") 

        if "file" not in request.files:
            print("⚠️ No file received!")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files.get("file")
        image_data = file.read()
        print(f"📏 Image Size: {len(image_data)} bytes")

        # Debugging step: Print file size
        print(f"Received image size: {len(image_data)} bytes")

        processed_image = preprocess_image(image_data)

        # Debugging step: Print shape after preprocessing
        print(f"Processed image shape: {processed_image.shape}")

        prediction = model.predict(processed_image)

        # Debugging step: Print model output
        print(f"Model Prediction: {prediction}")

        print(f"🔮 Prediction Output: {prediction.tolist()}")
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print(f"Error occurred: {e}")  # Print error in terminal
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)