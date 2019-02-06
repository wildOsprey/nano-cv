#Filtering img with sobel filter
import cv2
import numpy as np 
from .tools import *

gray = load_img('images//building.jpg')

show_img('Gray', gray, wait=False)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

filtered_image_x = cv2.filter2D(gray, -1, sobel_x)


sobel_y = np.array([[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]])

filtered_image_y= cv2.filter2D(gray, -1, sobel_y)
filtered_image = filtered_image_x + filtered_image_y

show_img('Filtered', filtered_image, wait=False)

retval, binary_img = cv2.threshold(filtered_image, 90, 255, cv2.THRESH_BINARY)
show_img('Binary', binary_img)
