import cv2
import numpy as np


def adaptive_threshold(image, window_size=15, C=5):
    offset = window_size // 2
    padded = np.pad(image, pad_width=offset, mode='constant', constant_values=0)
    adaptive = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+window_size, j:j+window_size]
            local_mean = np.mean(region)
            adaptive[i, j] = 255 if image[i, j] > local_mean - C else 0

    return adaptive

image = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)
adaptive = adaptive_threshold(image, 15, 5)

cv2.imshow("Adaptive Thresholding", adaptive)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

