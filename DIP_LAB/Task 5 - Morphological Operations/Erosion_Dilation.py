#Erosion and Dilation
import cv2
import numpy as np

def erosion(image, struct_ele):
    eroded_img = image.copy()
    height, width = image.shape
    struct_ele = struct_ele * 255
    offset = struct_ele.shape[0] // 2
    for r in range(height):
        for c in range(width):
            fit = True
            for x in range(-offset, offset+1):
                for y in range(-offset, offset+1):
                    if r+x >= 0 and r+x < height and c+y >= 0 and c+y < width:
                        sr, sc = x+offset, y+offset
                        if struct_ele[sr, sc] and image[r+x, c+y] != struct_ele[sr, sc]:
                            fit = False
                    elif (struct_ele[x+offset, y+offset]):
                        fit = False
            eroded_img[r, c] = 255 if fit else 0
    return np.uint8(eroded_img)

def dilation(image, struct_ele):
    dilated_img = image.copy()
    offset = struct_ele.shape[0] // 2
    height, width = image.shape
    struct_ele *= 255
    for r in range(height):
        for c in range(width):
            hit = False
            for x in range(-offset, offset+1):
                for y in range(-offset, offset+1):
                    sr, sc = x+offset, y+offset
                    if r+x >= 0 and r+x < height and c+y >= 0 and c+y < width:
                        if(image[r+x, c+y] and struct_ele[sr, sc] == struct_ele[sr, sc]):
                            hit = True
            dilated_img[r, c] = 255 if hit else 0
    return np.uint8(dilated_img)


img = cv2.imread('Noisy_Fingerprint.png', 0)
struct_ele_size = 3
struct_ele = np.ones((struct_ele_size, struct_ele_size))
eroded_img = erosion(img, struct_ele)
dilated_img = dilation(eroded_img, struct_ele)
cv2.imshow('Original Image', img)
cv2.imshow('Eroded Image', eroded_img)
cv2.imshow('Dilated Image', dilated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()