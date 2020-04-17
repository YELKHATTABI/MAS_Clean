import argparse

parser = argparse.ArgumentParser(description='This is a program to make predictions on images using VGG')
parser.add_argument("-w", "--weights_path",required=True,
                    help="path to model weights")
parser.add_argument("-i","--image_path",required=True,
                    help = "path to input image")
import cv2
from copy import deepcopy
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from ssd_keras.models.keras_ssd300 import ssd_300
from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_keras.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_keras.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from ssd_keras.keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from ssd_keras.keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_keras.ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from ssd_keras.data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_keras.data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_keras.data_generator.object_detection_2d_geometric_ops import Resize
from ssd_keras.data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# 1: Build the Keras model

args = parser.parse_args()

weights_path = args.weights_path
input_image_path = args.image_path

# Set the image size.
img_height = 300
img_width = 300

classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

colors_rgb = [
    (0,0,0),
    (247, 106, 123),
    (200, 101, 216),
    (184, 209, 112),
    (119, 246, 201),
    (78, 94, 100),
    (122, 254, 50),
    (118, 156, 187),
    (116, 131, 165),
    (222, 190, 167),
    (83, 90, 239),
    (169, 94, 209),
    (234, 90, 186),
    (255, 248, 54),
    (174, 111, 237),
    (98, 142, 225),
    (203, 225, 85),
    (194, 91, 186),
    (54, 88, 83),
    (168, 50, 142),
    (80, 243, 182),
]

def preprocess_image(image,target = (300,300)):
    """
    Return a preprocessed image as an array ready for prediction
    i.e : np array with shape (1,300,300,3)
    Argument : 
        image : path to the image to be preprocessed
        target : image size to be resized to 

    """
    image = cv2.resize(image, target)
    image = np.expand_dims(image, axis=0)
    return image

def render_image(original_image,y_pred_thresh,colors_rgb,classes,prediction_img_size = (300,300)):
    rendred_image = deepcopy(original_image)
    h,w,_ = original_image.shape
    img_height, img_width = prediction_img_size
    for box in y_pred_thresh[0]:
        xmin = int(box[2] * w / img_width)
        ymin = int(box[3] * h / img_height)
        xmax = int(box[4] * w / img_width)
        ymax = int(box[5] * h / img_height)
        pt1 = (xmin,ymin)
        pt2 = (xmax,ymax)
        color = colors_rgb[int(box[0])]
        rendred_image = cv2.rectangle(
            img=rendred_image,
            pt1=(xmin,ymin),
            pt2=(xmax,ymax),
            color=color,
            thickness=2,)
        label = "{}: {:.2f}".format(classes[int(box[0])], box[1])
        cv2.putText(
            rendred_image,
            label,
            (xmin+20,ymin+20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness=2
            )
    return rendred_image

def main():

    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)


    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.

    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    # model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    model.summary()

    image = cv2.imread(input_image_path)
    processed_image = preprocess_image(image,target = (300,300))
    y_pred = model.predict(processed_image)
    confidence_threshold = 0.8
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    rendered_image = render_image(image,y_pred_thresh,colors_rgb,classes)
    cv2.imshow('RenderedImage', rendered_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()