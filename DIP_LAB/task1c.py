#Task 1(c): Plot Histogram and Perform Single Threshold Segmentation

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Plot the histogram of the original grayscale image
plt.figure(figsize=(10, 5))

# Plot the grayscale image and its histogram side by side
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.hist(img.ravel(), bins=256, histtype='step', color='black')
plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()


threshold_value = 100 

_, img_thresholded = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images
cv2.imshow('Original Image', img)
cv2.imshow('Thresholded Image', img_thresholded)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
