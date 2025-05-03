import cv2
import numpy as np

def apply_ideal_lowpass_filter(image, cutoff_frequency):
    height, width = image.shape
    ideal_filter = np.zeros((height, width), dtype=np.float32)

    for u in range(height):
        for v in range(width):
            D = np.sqrt((u - height / 2)**2 + (v - width / 2)**2)
            ideal_filter[u, v] = 1 if D <= cutoff_frequency else 0

    # Apply FFT and filter
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    G_shifted = F_shifted * ideal_filter
    G = np.fft.ifftshift(G_shifted)
    filtered_image = np.fft.ifft2(G)

    return np.abs(filtered_image)


def apply_butterworth_filter(image, order, cutoff_frequency):
    height, width = image.shape
    butterworth_filter = np.zeros((height, width), dtype=np.float32)

    for u in range(height):
        for v in range(width):
            D = np.sqrt((u - height / 2)**2 + (v - width / 2)**2)
            butterworth_filter[u, v] = 1 / (1 + (D / cutoff_frequency)**(2 * order))

    # FFT of image
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    # Apply Butterworth filter
    G_shifted = F_shifted * butterworth_filter

    # Inverse FFT
    G = np.fft.ifftshift(G_shifted)
    filtered_image = np.fft.ifft2(G)

    return np.abs(filtered_image)

def apply_gaussian_lowpass_filter(image, cutoff_frequency):
    height, width = image.shape
    gaussian_filter = np.zeros((height, width), dtype=np.float32)

    for u in range(height):
        for v in range(width):
            D = np.sqrt((u - height / 2)**2 + (v - width / 2)**2)
            gaussian_filter[u, v] = np.exp(-(D**2) / (2 * (cutoff_frequency**2)))

    # Apply FFT and filter
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    G_shifted = F_shifted * gaussian_filter
    G = np.fft.ifftshift(G_shifted)
    filtered_image = np.fft.ifft2(G)

    return np.abs(filtered_image)

def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


img = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)
noisy_img = add_gaussian_noise(img, mean=0, std=15)

# Filter the noisy image
cutoff = 50
butter = apply_butterworth_filter(noisy_img, order=4, cutoff_frequency=cutoff)
gauss = apply_gaussian_lowpass_filter(noisy_img, cutoff_frequency=cutoff)
ideal = apply_ideal_lowpass_filter(noisy_img, cutoff_frequency=cutoff)

# Prepare display
cv2.imshow("Original", img)
cv2.imshow("Gaussian Noisy Image", noisy_img)
cv2.imshow("Butterworth Filtered", np.uint8(np.clip(butter, 0, 255)))
cv2.imshow("Gaussian Filtered", np.uint8(np.clip(gauss, 0, 255)))
cv2.imshow("Ideal Filtered", np.uint8(np.clip(ideal, 0, 255)))
cv2.waitKey(0)
cv2.destroyAllWindows()