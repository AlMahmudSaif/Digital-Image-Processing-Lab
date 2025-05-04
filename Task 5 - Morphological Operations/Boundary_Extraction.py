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

def extract_boundary(image, struct_ele):
    eroded_image = erosion(image.copy(), struct_ele)
    return image - eroded_image

img = cv2.imread('Binary_Image.png', 0)
struct_ele_size = 3
struct_ele = np.ones((struct_ele_size, struct_ele_size))
bound_img = extract_boundary(img, struct_ele)
cv2.imshow('Original Image', img)
cv2.imshow('Extracted Boundary', bound_img)
cv2.waitKey(0)
cv2.destroyAllWindows()