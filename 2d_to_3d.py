import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from scipy.spatial import cKDTree

# 📌 YOLO 모델 로드
model = YOLO("yolo12x.pt")

# 📌 이미지 로드 (캠 1: 측면, 캠 2-1: 후면)
image1_path = "./image1.jpg"  # 캠 1 (측면)
image2_path = "./image2.jpg"  # 캠 2-1 (후면)

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

if img1 is None or img2 is None:
    print("❌ 이미지 로드 실패")
    exit()

# 📌 YOLO를 이용하여 사람 검출 (바운딩 박스 시각화 포함)
def detect_people(image):
    results = model(image)
    person_data = []  # (cx, cy, w, h) 저장
    img_copy = image.copy()

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # 사람 클래스 ID = 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 중심점
                w, h = x2 - x1, y2 - y1  # 바운딩 박스 크기
                person_data.append((cx, cy, w, h))  # 좌표 + 크기 저장
                
                # 바운딩 박스 그리기
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)  # 중심점

    return np.array(person_data, dtype=np.float32), img_copy

# 📌 캠 1 (측면) & 캠 2-1 (후면)에서 사람 검출
people1, img1_viz = detect_people(img1)  # (cx, cy, w, h)
people2, img2_viz = detect_people(img2)

# 📌 검출된 사람이 1명 이상이어야 진행
if len(people1) < 1 or len(people2) < 1:
    print(f"❌ 검출된 사람이 부족함 (Image1: {len(people1)}, Image2: {len(people2)})")
    exit()

# 📌 좌표 기반 정렬을 활용한 매칭
def match_people(people1, people2):
    # X 좌표 기준 정렬 (측면 카메라)
    people1 = people1[np.argsort(people1[:, 0])]
    # Y 좌표 기준 정렬 (후면 카메라)
    people2 = people2[np.argsort(people2[:, 1])]

    matched_pts1, matched_pts2 = [], []

    min_len = min(len(people1), len(people2))
    for i in range(min_len):
        matched_pts1.append(people1[i][:2])
        matched_pts2.append(people2[i][:2])

    return np.array(matched_pts1), np.array(matched_pts2)

# 정렬 기반 매칭 적용
pts1, pts2 = match_people(people1, people2)

# 최소 1개 이상의 매칭이 필요
if len(pts1) < 1:
    print(f"❌ 충분한 매칭 포인트가 없음 (매칭된 개수: {len(pts1)})")
    exit()

# 📌 Fundamental Matrix 계산 (FM_RANSAC 추가)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

if F is None or F.shape != (3, 3):
    print("⚠️ FM_RANSAC 실패. FM_LMEDS 재시도...")
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# 최후의 방법: 기본 대각 행렬 생성
if F is None or F.shape != (3, 3):
    print("⚠️ F 강제 생성 (단순 대각행렬)")
    F = np.eye(3, dtype=np.float32) * 1e-5

# 📌 Essential Matrix 및 카메라 행렬 설정
K = np.array([[800, 0, 960],
              [0, 800, 540],
              [0, 0, 1]])

E = K.T @ F @ K  # 🚀 오류 발생 방지!

# 📌 3D 변환 수행
_, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)

P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, T))

points_4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3D = points_4D[:3] / points_4D[3]  # Homogeneous 좌표 변환

# 📌 이미지 & 3D 그래프 동시 출력
fig = plt.figure(figsize=(12, 6))

# 🔹 1. 검출된 이미지 표시
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(cv2.cvtColor(img1_viz, cv2.COLOR_BGR2RGB))
ax1.set_title("Image 1 - Detected People")
ax1.axis("off")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB))
ax2.set_title("Image 2 - Detected People")
ax2.axis("off")

# 🔹 2. 3D 복원된 사람 위치 그래프
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.scatter(pts1[:, 0], pts1[:, 1], zs=0, c='r', marker='o', label="People in Image 1")
ax3.scatter(pts2[:, 0], pts2[:, 1], zs=0, c='g', marker='x', label="People in Image 2")
ax3.scatter(points_3D[0], points_3D[1], points_3D[2], c='b', marker='o', label="Reconstructed 3D People")

ax3.set_xlabel("X axis")
ax3.set_ylabel("Y axis")
ax3.set_zlabel("Z axis")
ax3.set_title("3D People Localization (측면 & 후면)")
ax3.legend()

plt.tight_layout()
plt.show()
