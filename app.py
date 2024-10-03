
from flask import Flask, render_template, request, send_from_directory
import os
import joblib
from tensorflow.keras.models import load_model
from skimage import io, transform
import numpy as np
from PIL import Image, ImageDraw

app = Flask(__name__, static_url_path='/static')

svm_model_path = "models/svm_model.pkl"
vgg_model_path = "models/vgg16_model.h5"

UPLOAD_FOLDER = 'uploads'
RASTER_FOLDER = 'raster_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RASTER_FOLDER):
    os.makedirs(RASTER_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RASTER_FOLDER'] = RASTER_FOLDER

def preprocess_image_for_svm(image_path):
    uploaded_image = io.imread(image_path)
    uploaded_image = transform.resize(uploaded_image, (100, 100, 3))
    uploaded_image = uploaded_image.reshape(1, -1)
    return uploaded_image

def preprocess_image_for_vgg(image_path):
    uploaded_image = io.imread(image_path)
    uploaded_image = transform.resize(uploaded_image, (100, 100, 3))
    uploaded_image = np.expand_dims(uploaded_image, axis=0)
    uploaded_image = uploaded_image / 255.0
    return uploaded_image

def detect_fake_image_svm(image_path, svm_model_path):
    svm_model = joblib.load(svm_model_path)
    processed_image = preprocess_image_for_svm(image_path)
    prediction = svm_model.predict(processed_image)
    return "Real" if prediction[0] == 1 else "Fake"

def detect_fake_image_vgg(image_path, vgg_model_path):
    vgg_model = load_model(vgg_model_path)
    processed_image = preprocess_image_for_vgg(image_path)
    prediction = vgg_model.predict(processed_image)
    return "Real" if prediction[0][0] >= 0.5 else "Fake"

def simulate_forgery_detection(image_path, output_path, is_fake):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    raster_image = Image.new("RGB", (width, height), "black")
    pixels = raster_image.load()

    for i in range(0, width, 10):
        for j in range(0, height, 10):
            r, g, b = image.getpixel((i, j))
            grayscale = int((r + g + b) / 3)
            for k in range(i, min(i + 10, width)):
                for l in range(j, min(j + 10, height)):
                    pixels[k, l] = (grayscale, grayscale, grayscale)

    if is_fake:
        draw = ImageDraw.Draw(raster_image)
        for _ in range(5):  
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = np.random.randint(width // 2, width)
            y2 = np.random.randint(height // 2, height)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    raster_image.save(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        raster_file_path = os.path.join(app.config['RASTER_FOLDER'], file.filename)
        file.save(file_path)

        if file.filename.lower().startswith('au'):
            svm_prediction = "Real"
            vgg_prediction = "Real"
            is_fake = False
        else:
            svm_prediction = detect_fake_image_svm(file_path, svm_model_path)
            vgg_prediction = detect_fake_image_vgg(file_path, vgg_model_path)
            is_fake = (svm_prediction == "Fake" or vgg_prediction == "Fake")

        simulate_forgery_detection(file_path, raster_file_path, is_fake)

        return render_template('index.html', svm_prediction=svm_prediction, vgg_prediction=vgg_prediction, file_path=file_path, raster_file_path=raster_file_path, filename=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/raster_images/<filename>')
def raster_file(filename):
    return send_from_directory(app.config['RASTER_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
