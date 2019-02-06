#canny edge detection
import cv2
import numpy as np 
from .tools import *

gray = load_img('images//building.jpg')

canny_gray = cv2.Canny(gray, 250, 280)

show_img('high', canny_gray, wait = False)

canny_gray = cv2.Canny(gray, 50, 70)

show_img('low', canny_gray)