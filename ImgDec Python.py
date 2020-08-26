import os
import sys
import random
import cv2
import os
from pathlib import Path
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.model import log
import skimage.io

import PlantDetection

# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/balloon"):
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))
    # Go up two levels to the repo root
# Import Mask RCNN
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
PLANT_WEIGHTS_PATH = "plant2.h5"
config = PlantDetection.BalloonConfig()
PLANT_DIR = os.path.join(ROOT_DIR, "pictures/Plants")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
TEST_MODE = "inference"

#def get_ax(rows=1, cols=1, size=16):
    #"""Return a Matplotlib Axes array to be used in
    #all visualizations in the notebook. Provide a
    #central point to control graph sizes.

   # Adjust the size attribute to control how big to render images
    #"""
#    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
#    return ax

# Load validation dataset
dataset = PlantDetection.BalloonDataset()
dataset.load_balloon(PLANT_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config)

# Set path to balloon weights file
weights_path = PLANT_WEIGHTS_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

cam = cv2.VideoCapture(0)
cv2.startWindowThread()

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow('object detection', frame)
    img_name = "opencv_frame_{}.png".format(3)
    cv2.imwrite(img_name, frame)

    # Run object detection

    k = cv2.waitKey(1)
    if k%256 == 27:
        filename = "testpic.jpeg"
        image = skimage.io.imread(filename)
        results = model.detect([image], verbose=1)
    # Display result
        r = results[0]
        print(r['class_ids']-1)

        print(results)
        print(r['scores'])
        print(dataset.class_names)
        print("the start of the plant is at ", r['rois'][0][1] / 101, "cm")
        print("the end of the plant is at ", r['rois'][0][3] / 101, "cm")
    #if cv2.waitKey(20) == ord('q'):
        #break

cv2.destroyAllWindows()
del(cam)

