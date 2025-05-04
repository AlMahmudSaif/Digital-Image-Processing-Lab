import cv2
import numpy as np

def apply_filter(image, kernel):
    k = len(kernel) // 2
    padded = np.pad(image, pad_width=k, mode='constant', constant_values=0)
    result = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+2*k+1, j:j+2*k+1]
            result[i, j] = np.sum(region * kernel)
    return result

img = cv2.imread('image2.jpeg', 0).astype(float)

# Sobel kernels
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Prewitt kernels
prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Apply filters manually
sobel_x_res = apply_filter(img, sobel_x)
sobel_y_res = apply_filter(img, sobel_y)
sobel = np.sqrt(sobel_x_res**2 + sobel_y_res**2)

prewitt_x_res = apply_filter(img, prewitt_x)
prewitt_y_res = apply_filter(img, prewitt_y)
prewitt = np.sqrt(prewitt_x_res**2 + prewitt_y_res**2)

laplacian = np.abs(apply_filter(img, laplacian_kernel))

cv2.imshow("Sobel", sobel.astype(np.uint8))
cv2.imshow("Prewitt", prewitt.astype(np.uint8))
cv2.imshow("Laplacian", laplacian.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
