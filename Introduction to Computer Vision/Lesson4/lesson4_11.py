#k-means clustering

import numpy as np 
import cv2
from .tools import *

img = load_img('images//flamingos.jpg', colorful = True)

img = cv2.resize(img, (512, 512))
show_img('Monarch', img, wait = False)

# Reshape image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = img.reshape((-1,3))
print('Shape is', pixel_vals.shape)

# Convert to float type
pixel_vals = np.float32(pixel_vals)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 2
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((img.shape))
labels_reshape = labels.reshape(img.shape[0], img.shape[1])

show_img('Clustered', segmented_image, wait=False)


cluster = 0 # the first cluster

masked_image = np.copy(img)
# turn the mask green!
masked_image[labels_reshape == cluster] = [0, 255, 0]

show_img('Masked', masked_image)