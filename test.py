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


# 📌 Y축 간격 문제 해결을 위한 조정된 Homography 행렬
H1 = np.array([
    [0.0308, -0.0085, -12.92],  # ✅ X-Y 교차 영향 조정 (h12 증가)
    [0.0049, -0.035, 18.5],  # ✅ Y축 스케일 증가 (h22: -0.045 → -0.035)
    [0.0022, -0.012, 1.0]  # ✅ 원근 왜곡 보정 (h32: -0.015 → -0.012)
])

H2 = np.array([
    [0.0020, 0.0380, -6.26],  # ✅ X-Y 교차 영향 조정 (h12 감소)
    [0.0218, 0.0065, -4.5],  # ✅ Y축 스케일 감소 (h22: 0.004 → 0.0065)
    [0.0004, 0.007, 1.0]  # ✅ 원근 왜곡 보정 (h32: 0.0060 → 0.0055)
])

# 📌 YOLO를 이용하여 사람 검출 (박스 하단 중앙점 활용)
def detect_people(image):
    results = model(image)
    person_data = []  # (cx, y2) 저장
    img_copy = image.copy()

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # 사람 클래스 ID = 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                cx = (x1 + x2) // 2  # 중앙 X 좌표
                person_data.append([cx, y2])  # 하단 중앙점 저장
                
                # 바운딩 박스 및 하단 중앙점 시각화
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 박스
                cv2.circle(img_copy, (cx, y2), 5, (0, 0, 255), -1)  # 하단 중앙점 (발 위치)

    return np.array(person_data, dtype=np.float32), img_copy

# 📌 사람 검출 (박스 하단 중앙점 활용)
people1, img1_viz = detect_people(img1)  # Cam1 (측면)
people2, img2_viz = detect_people(img2)  # Cam2-1 (후면)

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
    axes[0].set_title("Image 1 - YOLO Detected (측면)")
    axes[0].axis("off")

    # 🔹 2. YOLO 검출된 이미지 2 (후면)
    axes[1].imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2 - YOLO Detected (후면)")
    axes[1].axis("off")

    # 🔹 3. Homography 적용된 2D 좌표 그래프
    axes[2].scatter(pts1[:, 0], pts1[:, 1], color='red', label="Transformed Cam1 (측면)", alpha=0.6)
    axes[2].scatter(pts2[:, 0], pts2[:, 1], color='blue', label="Transformed Cam2-1 (후면)", alpha=0.6)
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")
    axes[2].set_title("Homography 적용된 2D 좌표 (발 위치 기준)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# 📌 시각화 실행 (YOLO 검출 + Homography 변환된 좌표)
visualize_results(img1_viz, img2_viz, transformed_pts1, transformed_pts2)
