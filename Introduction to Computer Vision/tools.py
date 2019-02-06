import cv2
import numpy as np
def show_img(name, img, wait=True):
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey()

def show_imgs(name, split = False, *imgs):
    if split:
        for ind, img in enumerate(imgs):
            show_img('img'+str(ind), img, wait = False)
        cv2.waitKey()
    else:
        vis = np.concatenate(imgs, axis=1)
        vis = cv2.resize(vis, (1024, 768))
        show_img(name, vis)

def load_img(path, show=False, colorful = False):
    img = cv2.imread(path, int(colorful))
    if show:
        show_img('Image', img)
    return img

def RGB2HSV(img, show = False, views = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if views:
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]

        show_imgs('HSV', h, s, v)

    if show:
        show_img('HSV', img)
    return img
    
def BGR2RGB(img, show=False, views = False):
    img=  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if show:
        show_img('RGB', img)

    if views:
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        show_imgs('RGB', r, g, b)

    return img