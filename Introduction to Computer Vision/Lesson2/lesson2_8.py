import numpy as np 
import matplotlib.pyplot as plt 
import cv2

#smile 
def display_gray_image():
    image = cv2.imread('images/waymo_car.jpg')
    print('Image dimensions:', image.shape)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Image', gray_image)
    cv2.waitKey()

def display_smily_face():
    tiny_image = np.array([[0, 255, 0, 255, 0],
                      [0, 0, 0, 0, 0],
                      [255, 0, 0, 0, 255],
                      [0, 255, 50, 255, 0],
                      [0, 0, 255, 0, 0]])

    plt.matshow(tiny_image, cmap='gray')
    plt.show()

display_gray_image()

display_smily_face()