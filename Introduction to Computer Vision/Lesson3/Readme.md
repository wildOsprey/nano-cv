##Notes

### Filters

A low-pass filter (LPF) is a filter that passes signals with a frequency lower than a selected cutoff frequency and attenuates signals with frequencies higher than the cutoff frequency.  A high-pass filter (HPF) is the opposite to LPF. 

Frequency in images is a rate of change. Images change in space, and a high frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low frequency image may be one that is relatively uniform in brightness or changes very slowly.

You can extract frequency components with Fourier Transform (opencv). 

A common task for high filters is edge detection (sobel filters using cv2.filter2d() and a binary threshold for the better result).

Low-pass filters are used for blur/smoothing image, removing noise or block some high-frequency parts (GaussianBlur). 

High-pass and low-pass filters are often used together (ex. Canny Edge Detection).

For detection unifying boundaries of objects, you can use Hough Transfrom.

A good example of using filters for detection is Haar Cascades which is used for face detection task.