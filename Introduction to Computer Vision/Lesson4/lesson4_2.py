#harris corner detection
import cv2
import numpy as np
from .tools import *

img = load_img('images//waffle.jpg')

img_copy = np.copy(img)

show_img('Origin', img, wait=False)
img = np.float32(img)

dst = cv2.cornerHarris(img, 2, 3, 0.04) #3 - kernel, 2- corner filter

# Dilate corner image to enhance corner points
dst = cv2.dilate(dst,None)

show_img('Dilate', dst, wait = False)


thresh = 0.1*dst.max()

corner_image = np.copy(img_copy)

print('Dst', dst.shape)
# Iterate through all the corners and draw them on the image (if they pass the threshold)
for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if(dst[j,i] > thresh):
            # image, center pt, radius, color, thickness
            cv2.circle(corner_image, (i, j), 1, (0,255,0), 1)


show_img('Corners', corner_image, wait = True)