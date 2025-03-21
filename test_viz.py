import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 📌 YOLO 모델 로드
model = YOLO("head.pt")

# 📌 데이터 폴더 경로 설정
side_folder = "data/side/imgs"
back_folder = "data/back/imgs"

def extract_number(filename):
    """파일명에서 숫자 부분만 추출 (예: '144 (copy).jpg' → 144)"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # 숫자가 없는 경우 가장 뒤로 정렬

def load_images_in_order(folder_path):
    """폴더 내의 이미지를 순서대로 로드하고, 누락된 프레임은 스킵"""
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")], key=extract_number)
    images = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

# 이미지 불러오기
side_images = load_images_in_order(side_folder)
back_images = load_images_in_order(back_folder)

# 이미지가 순서대로 매칭되어야 하므로 최소한의 프레임 수를 기준으로 반복
num_frames = min(len(side_images), len(back_images))

# Homography 행렬 (예제 값, 필요시 변경)
H1 = np.array([
    [ 2.85314111e-01, -3.69907242e-01, -1.28672159e+02],
    [-4.28170215e-02, -7.87077601e-01,  5.67835561e+02],
    [-2.23331197e-04, -2.08590859e-03,  1.00000000e+00]
])

H2 = np.array([
    [-6.83579819e-01, -6.01078807e+00,  1.51252642e+03],
    [-2.64736625e+00, -1.58683687e+00,  1.62782158e+03],
    [-1.69854645e-03, -1.41089059e-02,  1.00000000e+00]
 ])

for i in range(num_frames):
    img1 = side_images[i]
    img2 = back_images[i]
    
    padding1 = (700,500)  # 측면 패딩 보정값
    padding2 = (300,150)  # 후면 패딩 보정값
    
    def detect_people(image, padding):
        results = model(image)
        person_data = []  # (cx, cy) 저장
        img_copy = image.copy()
        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    person_data.append([cx + padding[0], cy + padding[1]])
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
        return np.array(person_data, dtype=np.float32), img_copy

    people1, img1_viz = detect_people(img1, padding1)
    people2, img2_viz = detect_people(img2, padding2)
    
    if len(people1) < 1 or len(people2) < 1:
        print(f"❌ 검출된 사람이 부족함 (Frame {i})")
        continue

    def apply_homography(points, H):
        points_homogeneous = cv2.perspectiveTransform(np.array([points]), H)
        return points_homogeneous[0]

    transformed_pts1 = apply_homography(people1, H1)
    transformed_pts2 = apply_homography(people2, H2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img1_viz, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Side Frame {i} - YOLO Detected")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Back Frame {i} - YOLO Detected")
    axes[1].axis("off")
    
    axes[2].scatter(transformed_pts1[:, 0], transformed_pts1[:, 1], color='red', label="Transformed Cam1 (side)", alpha=0.6)
    axes[2].scatter(transformed_pts2[:, 0], transformed_pts2[:, 1], color='blue', label="Transformed Cam2-1 (back)", alpha=0.6)
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")
    axes[2].set_title("Homography 2D Coordinates")

    # ✅ 좌석 박스 추가 (0,0 기준 앞쪽 30 공백 후 시작)
    seat_width = 75  # 좌석 가로 크기
    seat_height = 50  # 좌석 세로 크기
    seat_start_x = 30 # 앞쪽 30 공백 유지
    seat_start_y = 0   # 좌석 시작 위치
    
    # 기존 4x2 좌석 배열
    for row in range(2):  # 총 2줄
        for col in range(4):  # 각 줄에 4개씩
            x = seat_start_x + col * seat_width
            y = seat_start_y + row * seat_height
            rect = plt.Rectangle((x, y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
            axes[2].add_patch(rect)

    # ✅ 추가된 좌석 (Y=240에서 시작, 3, 4번째 좌석)
    extra_seat_y = 190  # 새로운 좌석의 Y 시작 값
    extra_seat_x_1 = seat_start_x + 2 * seat_width  # 3번째 좌석 위치
    extra_seat_x_2 = seat_start_x + 3 * seat_width  # 4번째 좌석 위치
    
    rect1 = plt.Rectangle((extra_seat_x_1, extra_seat_y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
    rect2 = plt.Rectangle((extra_seat_x_2, extra_seat_y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
    
    axes[2].add_patch(rect1)
    axes[2].add_patch(rect2)

    axes[2].legend()
    axes[2].grid(True)
    
    plt.show()
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
