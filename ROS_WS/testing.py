import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread('2.jpg')

cv2.imshow("origin",image)
#cv2.waitKey()
# 이미지 크기 가져오기
height, width = image.shape[:2]

# 원본 이미지의 4개 코너 좌표 정의
# [좌측 상단, 우측 상단, 우측 하단, 좌측 하단]
src_points = np.float32([[0, 0], [width, 0], [465, 599], [160, 599]])


height = 599
width = 465 - 160
# 변환할 사다리꼴 모양의 좌표 정의
# 왼쪽 변이 길고 오른쪽 변이 짧은 사다리꼴
dst_points = np.float32([
    [0, 0],              # 좌측 상단 (x축 오른쪽으로 이동)
    [width, 0] ,      # 우측 상단 (x축 왼쪽으로 이동)
    [width, height], # 우측 하단 (x축 왼쪽으로 더 이동)
    [0, height]         # 좌측 하단 (x축 오른쪽으로 더 이동)
])

# 투시 변환 행렬 계산
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 이미지에 투시 변환 적용
transformed_image = cv2.warpPerspective(image, matrix, (width, height))


#transformed_image = cv2.resize(transformed_image , dsize=(600,600))
# 결과 이미지 표시
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)