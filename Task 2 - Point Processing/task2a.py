#2(a) Brightness Enhancement of Specific Range

import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

lower_bound = 150
upper_bound = 190

# Create a copy to modify
bright_img = img.copy()

for i in range(img.shape[0]): #img.shape[0] = height or rows of the image
    for j in range(img.shape[1]):
        if lower_bound <= img[i, j] <= upper_bound:
            bright_img[i, j] = min(img[i, j] + 20, 255) 

# Save and view the result
cv2.imshow('Original', img)
cv2.imshow('Brightness Enhanced Image', bright_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
