import cv2
import numpy as np

def sobel_filter(image):
    height, width = image.shape

    # 결과 이미지 (0으로 초기화)
    edge_image = np.zeros((height, width))

    # Sobel 커널 (official setting. x축 방향과 y축 방향으로 편미분을 수행하는 커널)
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 이미지를 순회하며 Sobel 필터 적용
    for y in range(1, height-1):
        for x in range(1, width-1):
            # x축 방향과 y축 방향의 그래디언트를 각각 계산
            Gx = np.sum(Kx * image[y-1:y+2, x-1:x+2]) # 커널과 이미지의 원소를 곱한 후 모두 더함(element-wise)
            Gy = np.sum(Ky * image[y-1:y+2, x-1:x+2]) # 커널과 이미지의 원소를 곱한 후 모두 더함(element-wise)
            # 해당 픽셀의 그래디언트 크기 계산
            edge_image[y, x] = np.sqrt(Gx**2 + Gy**2)
    
    edge_image = np.uint8(np.clip(edge_image / np.max(edge_image) * 255, 0, 255)) # 0~255 사이의 값으로 변환
    return edge_image

def unsharp_masking(image, sigma=3, strength=2): # sigma: 가우시안 블러의 표준편차, strength: 선명화 강도.
    height, width = image.shape
    # 가우시안 블러 처리한 이미지
    blurred_image = gaussian_blur(image, sigma)
    # Unsharp Masking 적용 (원본 이미지에서 가우시안 블러 처리한 이미지를 빼고 strength를 곱함. image=strength*(image-blurred_image)
    sharpened_image = np.clip(image + strength * (image - blurred_image), 0, 255) # 0~255 사이의 값으로 변환
    return np.uint8(sharpened_image)

def gaussian_blur(image, sigma):
    # size는 일반적으로 sigma의 3배 이상으로 설정(가우시안 분포의 99.7%가 해당 범위에 포함되기 때문). 대칭적인 커널이 되도록 2배 및 홀수로 설정
    size = int(sigma * 3) * 2 + 1
    # 가우시안 커널 생성
    gaussian_kernel = create_gaussian_kernel(size, sigma)
    height, width = image.shape
    blurred_image = np.zeros((height, width))
    # 가우시안 블러 적용 (가우시안 커널과 이미지의 원소를 곱한 후 모두 더함(element-wise))
    for y in range(size//2, height-size//2): # 이미지의 가장자리는 가우시안 블러 적용이 불가능하므로 제외
        for x in range(size//2, width-size//2):
            blurred_image[y, x] = np.sum(
                image[y-size//2:y+size//2+1, x-size//2:x+size//2+1] * gaussian_kernel) # 커널과 이미지의 원소를 곱한 후 모두 더함(element-wise)
            
    return blurred_image

def create_gaussian_kernel(size, sigma): # size: 커널의 크기, sigma: 가우시안 블러의 표준편차
    kernel = np.zeros((size, size))
    for y in range(size):
        for x in range(size):
            kernel[y, x] = np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2 * sigma**2)) # 가우시안 분포의 지수항. 표준편차가 클수록 0에 가까워짐
    kernel /= np.sum(kernel) # 커널의 모든 원소의 합이 1이 되도록 정규화
    return kernel

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
sobel_filtered = sobel_filter(image)
unsharp_filtered = unsharp_masking(image)

cv2.imshow('Sobel Filtered', sobel_filtered)
cv2.imshow('Unsharp Masking', unsharp_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()