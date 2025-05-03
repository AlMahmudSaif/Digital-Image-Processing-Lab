#Task 1(a): Decrease Spatial Resolution by Half

import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
resized = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation = cv2.INTER_AREA)
resized = cv2.resize(resized, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)

cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

