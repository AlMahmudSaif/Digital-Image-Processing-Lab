import cv2
import numpy as np


def histogram(image):
    temp = np.zeros(256, dtype = int)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            ok = image[i, j]
            temp[ok] += 1
    histo_img = np.full((200, 256), 255, dtype = np.uint8)
    max_val = np.max(temp)
    for x in range(256):
        hist_height = int((temp[x]/max_val) * 200)
        for y in range(200 - hist_height, 200):
            histo_img[y, x] = 0
    return histo_img

# Load grayscale image
image = cv2.imread('Binary_Image.png', cv2.IMREAD_GRAYSCALE)


cv2.imshow("Original Image", image)
cv2.imshow('Histrogram', histogram(image))
cv2.waitKey(0)
cv2.destroyAllWindows()
