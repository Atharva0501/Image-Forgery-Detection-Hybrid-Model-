
    uploaded_image = io.imread(image_path)
    uploaded_image = transform.resize(uploaded_image, (100, 100, 3))  # Resize image
    uploaded_image = np.expand_dims(uploaded_image, axis=0)
    uploaded_image = uploaded_image / 255.0  # Normalize pixel values
    # Predict using VGG16
    prediction = vgg_model.predict(uploaded_image)
    return "real" if prediction[0][0] >= 0.5 else "fake"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        svm_prediction = detect_fake_image_svm(file_path, svm_model_path)
        vgg_prediction = detect_fake_image_vgg(file_path, vgg_model_path)
        return render_template('index.html', svm_prediction=svm_prediction, vgg_prediction=vgg_prediction, file_path=file_path)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    return render_template('index.html', filename=uploaded_filename)

if __name__ == '__main__':
    app.run(debug=True)
