#orb rotation invariance (you can also try noise and illumination)
from tools import *
import numpy as np
#orb matching features
import cv2
import copy 
from orb import orb

img1 = load_img('Lesson5//images//face.jpeg')

img2 = load_img('Lesson5//images//faceR.jpeg')

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

keypoints_train, descriptors_train = orb(img1)
keypoints_query, descriptors_query = orb(img2)

matches = bf.match(descriptors_train, descriptors_query)

matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(img1, keypoints_train, img2, keypoints_query, matches[:300], img2, flags = 2)

show_img('Result', result)

print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))

print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))

print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
