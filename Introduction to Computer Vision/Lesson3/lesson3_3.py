#Fourier Transform
import cv2
import numpy as np 
from .tools import *

def apply_ft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))
    return frequency_tx

solid_image = load_img('images//diag_stripes.png', colorful = T rue)

#solid_image = BGR2RGB(solid_image)

solid_image = solid_image / 255.

solid_image_ft = apply_ft(solid_image)

show_imgs('Original and ft', solid_image, solid_image_ft)