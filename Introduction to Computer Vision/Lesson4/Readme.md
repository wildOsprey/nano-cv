## Notes

### Types of features and image segmentation

One of the common features is edges and corners. Harris Corner Detector is used to detect corners. The main idea is that it finds the difference in intensity for a displacement in all directions.

To reduce a noise from the image you can use dilation and erosion. Dilation enlarges bright, white areas in an image by adding pixels to the perceived boundaries of objects in that image. Erosion does the opposite: it removes pixels along object boundaries and shrinks the size of objects. You can use it in a pair to remove noise from the background (erosion, then dilation) which is called opening or to recover little blind spots from the object (dilation, then erosion) which is called closing. It is usually used on binary images.

Image segmentation task can be solved using countering techniques. Opencv provides build-in solution for this. 

K-means implementation is an algorithm for clustering. It takes k centers and divides area on k parts depends on these centers. Then it finds a mean for each part and moves center to it until the difference between mean and center smaller than a threshold. It can be used for image segmentation (cluster data by colors) or for the classification task.
