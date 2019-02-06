#Day and night classification based on brightness
import cv2
import numpy as np 
from .tools import *
from helpers import *

# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = load_dataset(image_dir_training)

STANDARDIZED_LIST = standardize(IMAGE_LIST)

import random

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = load_dataset(image_dir_test)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ', accuracy)
print("Number of misclassified images = " ,len(MISCLASSIFIED), ' out of '+ str(total))



def test_brightness():
    image_num = 0
    test_im = STANDARDIZED_LIST[image_num][0]

    avg = avg_brightness(test_im)
    print('Avg brightness: ' + str(avg))
    show_img('Test', test_im)

def test_load_standartize_estimate():
    image_index = 200
    selected_image = IMAGE_LIST[image_index][0]
    selected_label = IMAGE_LIST[image_index][1]
    print('Selected image shape', selected_image.shape)
    print('Selected image label', selected_label)
    # Display image and data about it
    show_img('Selected image', selected_image)

    print("Shape: "+str(selected_image.shape))
    print("Label: " + str(selected_label))

    # Select an image by index
    image_num = 0
    selected_image = STANDARDIZED_LIST[image_num][0]
    selected_label = STANDARDIZED_LIST[image_num][1]

    show_img('Selected', selected_image)
    print("Shape: "+str(selected_image.shape))
    print("Label [1 = day, 0 = night]: " + str(selected_label))

    print('Estimated label', estimate_label(selected_image))
