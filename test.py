import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 📌 YOLO 모델 로드
model = YOLO("yolo12x.pt")

# 📌 이미지 로드
image1_path = "./image1.jpg"  # Cam1 (측면)
image2_path = "./image2.jpg"  # Cam2-1 (후면)

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

if img1 is None or img2 is None:
    print("❌ 이미지 로드 실패")
    exit()

# 패딩 추가
padding1 = 500 # 측면 500px 패딩 추가
padding2 = 150 # 후면 150px 패딩 추가

# 📌 Y축 간격 문제 해결을 위한 조정된 Homography 행렬
H1 = np.array([
    [ 2.35934416e-01, -3.07335620e-01, -3.40956275e+01],
    [-4.16231569e-02, -8.26516973e-01,  6.04035253e+02],
    [-4.49255396e-04, -1.71300269e-03,  1.00000000e+00]
])

H2 = np.array([
    [ 4.98517970e+00,  3.97303716e+01, -1.08080207e+04],
    [ 2.39892969e+01,  1.54180682e+01, -1.15825701e+04],
    [ 1.48942095e-02,  7.23936521e-02,  1.00000000e+00]
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
                person_data.append([cx + padding, cy + padding])  # 패딩을 고려한 중앙점 저장
                
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

# 📌 Homography 변환 적용 (박스 하단 중앙점 활용)
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
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# 📌 시각화 실행 (YOLO 검출 + Homography 변환된 좌표)
visualize_results(img1_viz, img2_viz, transformed_pts1, transformed_pts2)
