#dilation and erosion 
import cv2
import numpy as np 
from .tools import *

# Reads in a binary image
image = load_img('images//letter.jpg')

show_img('Origin', image)

# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)

# Dilate the image
dilation = cv2.dilate(image, kernel, iterations = 1)

show_img('Dilation', dilation)

erosion = cv2.erode(image, kernel, iterations = 1)

show_img('Erosion', erosion)

#opening and closing
j_noise = load_img('images//j_noise.jpg')
opening = cv2.morphologyEx(j_noise, cv2.MORPH_OPEN, kernel)
show_img('Opening', opening)

j_halls = load_img('images//j_halls.jpg')
closing = cv2.morphologyEx(j_halls, cv2.MORPH_CLOSE, kernel)
show_img('Closing', closing)
