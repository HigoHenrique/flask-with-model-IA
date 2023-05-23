from flask import Flask, request
from flask import render_template
import numpy as np
import os
from uuid import uuid4
from PIL import Image
from keras.preprocessing import image

from keras.models import load_model

app = Flask(__name__)
model = load_model('modelo-cat-dog.h5')

app.config['UPLOAD_FOLDER'] = 'static'


def preprocess(img):
    img_resized = img.resize((150, 150))
    img_array = image.image_utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


def get_class_info(img_processed):
    prediction = model.predict(img_processed)
    if prediction < 0.5:
        return "Prediction: Cat (%.2f%%)" % (100 - prediction[0][0] * 100)
    else:
        return "Prediction: Dog (%.2f%%)" % (prediction[0][0] * 100)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/resultado')
def resultado():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        ############
        _, file_extension = os.path.splitext(file.filename)
        filename = f'{uuid4().hex}{file_extension}'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_path = image_path.replace('\\', '/')
        ###########
        img = Image.open(image_path).convert("RGB")
        img_processed = preprocess(img)
        prediction = get_class_info(img_processed)
        return render_template("resultado.html", resultado=prediction, image_path=image_path)
    else:
        return render_template("resultado.html", resultado="Imagem não enviada")


if __name__ == '__main__':
    app.run(debug=True)
