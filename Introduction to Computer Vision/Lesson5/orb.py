import numpy as np
import cv2
import copy

def orb(img, draw = False):
    orb = cv2.ORB_create(1000, 2.0)

    keypoints, descriptor = orb.detectAndCompute(img, None)

    keyp_without_size = copy.copy(img)
    keyp_with_size = copy.copy(img)
    if draw:
        keyp_without_size = cv2.drawKeypoints(img, keypoints, keyp_without_size, color = (0, 255, 0))
        keyp_with_size = cv2.drawKeypoints(img, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return keypoints, descriptor, keyp_without_size, keyp_with_size
    
    return keypoints, descriptor