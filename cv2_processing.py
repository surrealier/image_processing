import cv2
import numpy as np

def apply_sobel(image, ksize=3):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)

    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel / np.max(sobel) * 255, 0, 255))
    return sobel

def apply_unsharp_masking(image, sigma=2.0, strength=5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def apply_canny(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)

def apply_prewitt(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(image, -1, kernelx)
    prewitty = cv2.filter2D(image, -1, kernely)
    prewitt = np.sqrt(prewittx**2 + prewitty**2)
    prewitt = np.uint8(np.clip(prewitt / np.max(prewitt) * 255, 0, 255))
    return prewitt

def apply_log(image, sigma=1.0):
    log = cv2.GaussianBlur(image, (0, 0), sigma)
    log = cv2.Laplacian(log, cv2.CV_64F)
    return log

# image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('Lenna.png')
cv2.imshow('Lenna', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 필터 적용
sobel_filtered = apply_sobel(image)
unsharp_filtered = apply_unsharp_masking(image)
canny_filtered = apply_canny(image)
prewitt_filtered = apply_prewitt(image)
log_filtered = apply_log(image)

# 결과 출력
cv2.imshow('Sobel Filtered', sobel_filtered)
cv2.imshow('Unsharp Masking', unsharp_filtered)
cv2.imshow('Canny Filtered', canny_filtered)
cv2.imshow('Prewitt Filtered', prewitt_filtered)
cv2.imshow('LoG Filtered', log_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
