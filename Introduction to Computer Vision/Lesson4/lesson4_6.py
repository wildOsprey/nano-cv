#contours

import cv2
import numpy as np 
from .tools import *

def orientations(contours):
    """
    Orientation 
    :param contours: a list of contours
    :return: angles, the orientations of the contours
    """
    
    angles = []
    for contour in contours:
        (x,y), (MA,ma), angle = cv2.fitEllipse(contour)
        angles.append(contour)
    return angles

def left_hand_crop(image, selected_contour):
    cropped_image = np.copy(image)

    # Find the bounding rectangle of a selected contour
    x,y,w,h = cv2.boundingRect(selected_contour)
    
    # Crop the image using the dimensions of the bounding rectangle
    cropped_image = cropped_image[y: y + h, x: x + w]
    
    return cropped_image
    
    return cropped_image
img = load_img('images//thumbs_up_down.jpg')

show_img('Origin', img)

retval, binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)

show_img('Binary', binary)

# Find contours from thresholded, binary image
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on a copy of the original image
contours_image = np.copy(img)
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)

show_img('Contours', contours_image)

angles = orientations(contours)

selected_contour = contours[1]


if(selected_contour is not None):
    cropped_image = left_hand_crop(img, selected_contour)
    show_img('Selected', cropped_image)