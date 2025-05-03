import cv2
import numpy as np

def calculate_psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return 100
    else:
        pixel_max = 255.0
        return 20 * np.log10(pixel_max / np.sqrt(mse))

def apply_harmonic_mean_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape 
    offset, number_of_pixel = mask_size // 2, mask_size * mask_size

    for r in range(height):
        for c in range(width):
            pixel = 0
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        pixel += float(1 / (image[r + x, c + y] + 1e-4))
            pixel = number_of_pixel / pixel
            filtered_image[r, c] = 255 if pixel > 255 else pixel

    return np.uint8(filtered_image)

def apply_geometric_mean_filter(image, mask_size):
    filtered_image = image.copy()
    height, width = filtered_image.shape
    offset = mask_size // 2

    for r in range(height):
        for c in range(width):
            pixel = 1
            count = 0
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if (r + x >= 0 and r + x < height and c + y >= 0 and c + y < width):
                        if (image[r + x, c + y]):
                            count += 1
                            pixel = pixel * int(image[r + x, c + y])
            count = 1 if count == 0 else count
            filtered_image[r, c] = pixel ** (1 / count)

    return np.uint8(filtered_image)

img = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)
noisy_img = img.copy()

prob = 0.05
th = 1-prob

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rnd = np.random.rand()
        if(rnd < prob):
            noisy_img[i, j] = 0
        elif rnd > th:
            noisy_img[i,j] = 255

harmonic_img = apply_harmonic_mean_filter(noisy_img, 3)
geometric_img = apply_geometric_mean_filter(noisy_img, 3)

harmonic_psnr = calculate_psnr(img, harmonic_img)
geometric_psnr = calculate_psnr(img, geometric_img)

print(f'PSNR after Harmonic Mean Filter: {harmonic_psnr:.2f} dB')
print(f'PSNR after Geometric Mean Filter: {geometric_psnr:.2f} dB')

cv2.imshow('Original Image', img)
cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Harmonic Filtered', harmonic_img)
cv2.imshow('Geometric Filtered', geometric_img)
cv2.waitKey(0)
cv2.destroyAllWindows()