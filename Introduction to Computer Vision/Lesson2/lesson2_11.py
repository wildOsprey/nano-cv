import numpy as np 
import matplotlib.pyplot as plt 
import cv2

#RGB vs BGR

def display_R_G_B_image():
    image = cv2.imread('images/waymo_car.jpg')
    print('Image dimensions:', image.shape)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Isolate RGB channels
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]

    cv2.imshow('R', r)
    cv2.imshow('G', g)
    cv2.imshow('B', b)
    cv2.waitKey()


display_R_G_B_image()