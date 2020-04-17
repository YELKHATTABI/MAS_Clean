import argparse

parser = argparse.ArgumentParser(description='This is a program to make predictions on images using VGG')
parser.add_argument("-w", "--weights_path",required=True,
                    help="path to model weights")
parser.add_argument("-i","--image_path",required=True,
                    help = "path to input image")
from tensorflow.keras.models import load_model
import cv2
import numpy as np 


args = parser.parse_args()

model_path = args.weights_path
input_image_path = args.image_path


def read_preprocess_image(image_path):
    """
    Return a preprocessed image as an array ready for prediction
    i.e : np array with shape (1,224,224,3)
    Argument : 
        image_path : path to the image to be preprocessed

    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224))
    image = np.expand_dims(image, axis=0)
    return image

    
# Load the model

VGG16 = load_model(model_path)

# read input image
image = read_preprocess_image(image_path=input_image_path)

# make prediction
prediction = VGG16.predict(image)
print( F"""
Model prediction for the image {input_image_path} using VGG16 pretrained to classify Dogs and Cats ***
Dog : {100*prediction[0][0]:.2f} %
Cat : {100*prediction[0][1]:.2f} %
"""
)