#Task 1(b): Decrease Intensity Level Resolution by 1 Bit

import cv2

# Load grayscale image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Decrease intensity by 1 bit
img_binary = (img >> 1) 

# Show results
cv2.imshow('Original', img)
cv2.imshow('1-bit Intensity (Binary)', img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
