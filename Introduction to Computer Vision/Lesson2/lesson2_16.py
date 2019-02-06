import numpy as np
import matplotlib.pyplot as plt
import cv2
from .tools import *
#Choose pink balloons from the pict

img = load_img('images/balloons.jpg', show=True)

img = BGR2RGB(img)

# RGB channels
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

show_imgs('RGB', r, g, b)

# Convert from RGB to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

show_imgs('HSV', h, s, v)

# Define our color selection criteria in HSV values
lower_hue = np.array([160,0,0]) 
upper_hue = np.array([180,255,255])

# Define our color selection criteria in RGB values
lower_pink = np.array([180,0,100]) 
upper_pink = np.array([255,255,230])

# Define the masked area in RGB space
mask_rgb = cv2.inRange(img, lower_pink, upper_pink)

# mask the image
masked_image = np.copy(img)
masked_image[mask_rgb==0] = [0,0,0]

# Vizualize the mask
show_img('Masked image', masked_image)

# Define the masked area in HSV space
mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)

# mask the image
masked_image = np.copy(img)
masked_image[mask_hsv==0] = [0,0,0]

masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
# Vizualize the mask
show_img('HSV mask', masked_image)
