#orb finding features
from tools import *
import numpy as np
import cv2
import copy 

img = load_img('Lesson5//images//face.jpeg')

orb = cv2.ORB_create(200, 2.0)

keypoints, descriptor = orb.detectAndCompute(img, None)

keyp_without_size = copy.copy(img)
keyp_with_size = copy.copy(img)

keyp_without_size = cv2.drawKeypoints(img, keypoints, keyp_without_size, color = (0, 255, 0))

keyp_with_size = cv2.drawKeypoints(img, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_img('Keypoints', keyp_without_size, wait = False)
show_img('Keypoints with size', keyp_with_size)

print("\nNumber of keypoints Detected: ", len(keypoints))