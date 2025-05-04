#Opening and Closing
import cv2
import numpy as np

def erosion(image, struct_ele):
    eroded_img = image.copy()
    height, width = image.shape
    ksize = struct_ele.shape[0]
    offset = ksize // 2
    for r in range(height):
        for c in range(width):
            min_val = 255
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if 0 <= r + x < height and 0 <= c + y < width:
                        min_val = min(min_val, image[r + x, c + y])
            eroded_img[r, c] = min_val
    return np.uint8(eroded_img)

def dilation(image, struct_ele):
    dilated_img = image.copy()
    height, width = image.shape
    ksize = struct_ele.shape[0]
    offset = ksize // 2
    for r in range(height):
        for c in range(width):
            max_val = 0
            for x in range(-offset, offset + 1):
                for y in range(-offset, offset + 1):
                    if 0 <= r + x < height and 0 <= c + y < width:
                        max_val = max(max_val, image[r + x, c + y])
            dilated_img[r, c] = max_val
    return np.uint8(dilated_img)


def opening(image, struct_ele):
    return dilation(erosion(image, struct_ele), struct_ele)

def closing(image, struct_ele):
    return erosion(dilation(image, struct_ele), struct_ele)


img = cv2.imread('Noisy_Fingerprint.png', 0)
struct_ele_size = 3
struct_ele = np.ones((struct_ele_size, struct_ele_size))
opened_img = opening(img, struct_ele)
closed_img = closing(opened_img, struct_ele)
cv2.imshow('Original Image', img)
cv2.imshow('Opened Image', opened_img)
cv2.imshow('Closed Image', closed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

