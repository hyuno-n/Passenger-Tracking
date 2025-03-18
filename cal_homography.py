import cv2
import numpy as np

def add_padding(image, padding=150, color=(0, 0, 0)):
    """이미지 상하좌우에 패딩 추가"""
    h, w = image.shape[:2]
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                      cv2.BORDER_CONSTANT, value=color)
    return padded_image

# 이미지 로드
img1 = cv2.imread("image1.jpg")  # Cam1 (측면)
img2 = cv2.imread("image2.jpg")  # Cam2-1 (후면)

# 패딩 추가
img1_padded = add_padding(img1, padding=150)  # 150px 패딩 추가
img2_padded = add_padding(img2, padding=150)

# 변환된 이미지 확인
cv2.imshow("Padded Image 1", img1_padded)
cv2.imshow("Padded Image 2", img2_padded)
cv2.waitKey(0)
cv2.destroyAllWindows()
