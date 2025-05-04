# 3.(a) Apply average and median spatial filters with 5×5 mask 
# and observe their performance for noise suppression in terms of PSNR.

import cv2
import numpy as np

def average_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape
    offset, weight = mask_size // 2, mask_size * mask_size

    for r in range(height):
        for c in range(width):
            filtered_image[r, c] = 0;
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        filtered_image[r, c] += (image[r + x, c + y] / weight)
    
    return np.uint8(filtered_image)

def median_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape
    offset = mask_size // 2

    for r in range(height):
        for c in range(width):
            pixels = []
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        pixels.append(image[r + x, c + y])
            filtered_image[r, c] = sorted(pixels)[len(pixels) // 2]
    
    return np.uint8(filtered_image)

def find_psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    pixel_max = 255
    return 20 * np.log10(pixel_max / np.sqrt(mse))

img = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)

noisy_img = img.copy()
prob = 0.05
th = 1-prob
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rnd = np.random.rand()
        if rnd < prob:
            noisy_img[i, j] = 0
        elif rnd > th:
            noisy_img[i, j] = 255

avg_img = average_filter(noisy_img, 5)
median_img = median_filter(noisy_img, 5)

psnr_avg = find_psnr(img, avg_img)
psnr_median = find_psnr(img, median_img)

print(f"PSNR of Average Filtered Image: {psnr_avg}")
print(f"PSNR of Median Filtered Image: {psnr_median}")

cv2.imshow('Original Image', img)
cv2.imshow('Average Filtered', np.uint8(avg_img))
cv2.imshow('Median Filterred', np.uint8(median_img))
cv2.imshow('Noisy Image', noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
