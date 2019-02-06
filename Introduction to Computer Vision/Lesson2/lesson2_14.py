import cv2
import numpy as np 
import matplotlib.pyplot as plt 

#crop image from r g b background and replace with a new background

def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()

#load original image
img = cv2.imread('images//girl.jpg')
show_img('img BGR', img)

#convert to RGB
#already in RGB?
img_copy = np.copy(img)
#img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
#show_img('img RGB', img_copy)

#define threshold
lower_green = np.array([0,230,0])
upper_green = np.array([70,255,70])
mask = cv2.inRange(img_copy, lower_green, upper_green)
show_img('img MASK', mask)

#create mask
masked_image = np.copy(img_copy)
masked_image[mask != 0] = [0, 0, 0]
show_img('img with MASK', masked_image)

#create background mask
background = cv2.imread('images//space.jpg')
background = background[0:img.shape[0], 0:img.shape[1]]
background[mask == 0] = [0, 0, 0]
show_img('background with MASK', background)

#create complete pic
img_with_background = masked_image + background
show_img('image with background', img_with_background)
