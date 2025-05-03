# 3.(b) Use different sizes of mask (3×3, 5×5, 7×7) with average filter for noise suppression
# and observe their performance in terms of PSNR.

import cv2
import numpy as np

def average_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape
    offset, weight = mask_size // 2, mask_size * mask_size

    for r in range(height):
        for c in range(width):
            filtered_image[r, c] = 0
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        filtered_image[r, c] += (image[r + x, c + y] / weight)
    
    return np.uint8(filtered_image)


def calculate_psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))



img = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)
noisy_img = img.copy()
prob = 0.05
th = 1-prob
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rnd = np.random.rand()
        if rnd < prob:
            noisy_img[i, j] = 1
        elif rnd > th:
            noisy_img[i,j] = 254

mask_3x3 = average_filter(noisy_img, 3)
mask_5x5 = average_filter(noisy_img, 5)
mask_7x7 = average_filter(noisy_img, 7)

psnr_3x3 = calculate_psnr(img, mask_3x3)
psnr_5x5 = calculate_psnr(img, mask_5x5)
psnr_7x7 = calculate_psnr(img, mask_7x7)

print(f'PSNR (3x3 average filter): {psnr_3x3:.2f} dB')
print(f'PSNR (5x5 average filter): {psnr_5x5:.2f} dB')
print(f'PSNR (7x7 average filter): {psnr_7x7:.2f} dB')

cv2.imshow('Mask size 3x3', mask_3x3)
cv2.imshow('Mask size 5x5', mask_5x5)
cv2.imshow('Mask size 7x7', mask_7x7)
cv2.imshow('Original Image', img)
cv2.imshow('Noisy Image', noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
