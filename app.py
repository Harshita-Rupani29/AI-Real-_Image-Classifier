

from flask import Flask, render_template, request
import os
import cv2 as cv
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model("AIGeneratedModel.h5")

# Function to detect if the image is AI-generated or real
def detect_image_type(image_path):
    img_arr = cv.imread(image_path)
    new_arr = cv.resize(img_arr, (48, 48)) / 255.0
    test = np.array([new_arr])
    result = model.predict(test)
    return "AI Generated" if result[0][0] > 0.5 else "Real"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    filename = file.filename
    file.save(os.path.join('static', 'uploads', filename))
    image_path = os.path.join('static', 'uploads', filename)
    image_type = detect_image_type(image_path)
    return f'The given image is: {image_type}'

if __name__ == "__main__":
    app.run(debug=True)
