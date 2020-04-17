import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer


app = Flask(__name__)


def get_model():
    global model
    model = load_model(
        os.path.expandvars("$HOME/MAS_Clean/VGG_16_cat_and_dogs/model_weights/VGG16_cats_and_dogs.h5")
    )
    print("* Model is loaded !")


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    decoded_image = cv2.imdecode(img_array, cv2_img_flag)

    return decoded_image

def preprocess_image(image):
    """
    Return a preprocessed image as an array ready for prediction
    i.e : np array with shape (1,224,224,3)
    Argument : 
        image_path : path to the image to be preprocessed

    """
    image = cv2.resize(image, (224,224))
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras model...")
get_model()
print(F"""
Use the following link to access the app : http://0.0.0.0:5000/static/predict_classification.html
""")


@app.route("/predict_classification", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded_image = message["image"]
    decoded_image = base64.b64decode(encoded_image)
    image = create_opencv_image_from_stringio(io.BytesIO(decoded_image))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image).tolist()
    response = {"prediction": {"dog": prediction[0][0], "cat": prediction[0][1]}}
    return jsonify(response)


if __name__ == "__main__":
    # Serve the app with gevent
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
