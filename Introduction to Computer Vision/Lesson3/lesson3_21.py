#hough lines

import cv2
import numpy as np 
from .tools import *

gray = load_img('images//phone.jpg')

low_threshold = 50
high_threshold = 100

edges = cv2.Canny(gray, low_threshold, high_threshold)
show_img('Canny phone', edges)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 50
max_line_gap = 5

line_image = np.copy(gray) #creating an image copy to draw lines on

# Run Hough on the edge-detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

                    
# Iterate over the output "lines" and draw lines on the image copy
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

show_img('Found', line_image)

#hough circles
# for drawing circles on

gray = load_img('images//round_farms.jpg')

gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

show_img('Blur',gray_blur)

circles_im = np.copy(gray)

circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 
                           minDist=45,
                           param1=70,
                           param2=11,
                           minRadius=20,
                           maxRadius=40)

# convert circles into expected type
circles = np.uint16(np.around(circles))
# draw each one
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(circles_im,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(circles_im,(i[0],i[1]),2,(0,0,255),3)
    
show_img('Circle', circles_im)

print('Circles shape: ', circles.shape)