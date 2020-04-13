import base64
import numpy as np
import io
from PIL import Image
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
        "/Users/yelkhattabi/MAS_Clean/VGG_16_weights/VGG16_cats_and_dogs.h5"
    )
    model._make_predict_function()
    print("* Model is loaded !")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


print(" * Loading Keras model...")
get_model()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded_image = message["image"]
    decoded_image = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(decoded_image))
    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image).tolist()

    response = {"prediction": {"dog": prediction[0][0], "cat": prediction[0][1]}}
    return jsonify(response)


if __name__ == "__main__":
    # Serve the app with gevent
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
