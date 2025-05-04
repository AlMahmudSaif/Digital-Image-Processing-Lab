#2(b) Differentiate the results of power law and inverse logarithmic transform

import cv2
import numpy as np

img = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)

# Power Law Transform (Gamma Correction)
gamma = 0.5
normalized = img / 255.0
power_img = np.uint8((normalized ** gamma) * 255) 

# Inverse Logarithmic Transform
c = 255 / np.log(256)
inv_log_img = np.uint8(c * np.exp(img / c) - 1)

# Show the results
cv2.imshow('Original', img)
cv2.imshow('Power Law Transform', power_img)
cv2.imshow('Inverse Logarithmic Transform', inv_log_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
