#img in different scales
from tools import *
import numpy as np
import cv2

img = load_img('Lesson5//images//rainbow_flag.jpg', colorful = True)

show_img('Origin', img, wait = False)

level_1 = cv2.pyrDown(img)
level_2 = cv2.pyrDown(level_1)
level_3 = cv2.pyrDown(level_2)

show_imgs('Pyramid', True, level_1, level_2, level_3)