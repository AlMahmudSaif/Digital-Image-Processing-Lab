#2(c) Find difference image between original and MSB image (last three bits)

import cv2
import numpy as np

img = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)

# Extract the Most Significant 3 Bits (MSB)
msb_img = img & 0b11100000

# Find absolute difference
diff_img = cv2.absdiff(img, msb_img)

# Show the results
cv2.imshow('Original Image', img)
cv2.imshow('MSB Image (Last 3 Bits)', msb_img)
cv2.imshow('Difference Image', diff_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
