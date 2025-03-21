import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 📌 YOLO 모델 로드
model = YOLO("head.pt")

# 📌 이미지 로드
image1_path = "./image1.jpg"  # Cam1 (측면)
image2_path = "./image2.jpg"  # Cam2-1 (후면)

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

if img1 is None or img2 is None:
    print("❌ 이미지 로드 실패")
    exit()

# 패딩 추가
padding1 = (700,500) # 측면 패딩 보정값
padding2 = (300,150) # 후면 패딩 보정값

# 📌 Y축 간격 문제 해결을 위한 조정된 Homography 행렬
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

# 📌 YOLO를 이용하여 사람 검출 (박스 중앙점 활용)
def detect_people(image, padding):
    results = model(image)
    person_data = []  # (cx, cy) 저장
    img_copy = image.copy()

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # 사람 클래스 ID = 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                cx = (x1 + x2) // 2  # 중앙 X 좌표
                cy = (y1 + y2) // 2  # 중앙 Y 좌표
                person_data.append([cx + padding[0], cy + padding[1]])  # 패딩을 고려한 중앙점 저장
                
                # 바운딩 박스 및 중앙점 시각화
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 박스
                cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)  # 중앙점

    return np.array(person_data, dtype=np.float32), img_copy

# 📌 사람 검출 (박스 하단 중앙점 활용, 패딩 보정 추가)
people1, img1_viz = detect_people(img1, padding1)  # Cam1 (측면)
people2, img2_viz = detect_people(img2, padding2)  # Cam2-1 (후면)

if len(people1) < 1 or len(people2) < 1:
    print(f"❌ 검출된 사람이 부족함 (Image1: {len(people1)}, Image2: {len(people2)})")
    exit()

# 📌 Homography 변환 적용 
def apply_homography(points, H):
    points_homogeneous = cv2.perspectiveTransform(np.array([points]), H)
    return points_homogeneous[0]

transformed_pts1 = apply_homography(people1, H1)
transformed_pts2 = apply_homography(people2, H2)

# 📌 YOLO 검출 결과 & Homography 변환된 좌표 시각화
def visualize_results(img1_viz, img2_viz, pts1, pts2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 🔹 1. YOLO 검출된 이미지 1 (측면)
    axes[0].imshow(cv2.cvtColor(img1_viz, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image 1 - YOLO Detected (side)")
    axes[0].axis("off")

    # 🔹 2. YOLO 검출된 이미지 2 (후면)
    axes[1].imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2 - YOLO Detected (back)")
    axes[1].axis("off")

    # 🔹 3. Homography 적용된 2D 좌표 그래프
    axes[2].scatter(pts1[:, 0], pts1[:, 1], color='red', label="Transformed Cam1 (side)", alpha=0.6)
    axes[2].scatter(pts2[:, 0], pts2[:, 1], color='blue', label="Transformed Cam2-1 (back)", alpha=0.6)
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")
    axes[2].set_title("Homography 2D Coordinates")
    
    # ✅ 좌석 박스 추가 (0,0 기준 앞쪽 30 공백 후 시작)
    seat_width = 75  # 좌석 가로 크기
    seat_height = 50  # 좌석 세로 크기
    seat_start_x = 15  # 앞쪽 30 공백 유지
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

    plt.tight_layout()
    plt.show()



# 📌 시각화 실행 (YOLO 검출 + Homography 변환된 좌표)
visualize_results(img1_viz, img2_viz, transformed_pts1, transformed_pts2)
