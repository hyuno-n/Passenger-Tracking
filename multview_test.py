import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------- 이미지 로드 --------------------
img1 = cv2.imread("frame_rear.jpg")
img2 = cv2.imread("frame_center.jpg")
img3 = cv2.imread("frame_front.jpg")

# -------------------- YOLO 모델 로드 --------------------
model = YOLO("head.pt")

# -------------------- Homography 행렬 수동 설정 --------------------
# 📌 Homography Matrix (Image 1): # 후면(angle 40) padding (500,150)
H1 = np.array([[ 3.39218386e-01, -2.61378020e+00, -2.41384154e+02],
               [ 1.23077482e+00, -1.09011484e+00, -8.55740148e+02],
               [ 8.33939613e-04, -9.20352638e-03,  1.00000000e+00]])
# 📌 Homography Matrix (Image 2): # 중앙(angle 0) padding (300,150)
H2 = np.array([[ -1.84509850e-01,  8.03468203e-02,  5.25063189e+02],
                [ 4.81525443e-02,  3.72219168e-01, -8.28806408e+01],
                [ 2.24470429e-04,  2.05735101e-04,  1.00000000e+00]])
# 📌 Homography Matrix (Image 3): # 전면(angle -40) padding (300,150)
H3 = np.array([[ -5.95639903e-01, -4.92610298e+00,  1.36820095e+03],
                [-1.89307888e+00, -1.34889107e+00,  1.32959556e+03],
                [-1.56119349e-03, -1.07253880e-02,  1.00000000e+00]])

# -------------------- 패딩 정보 --------------------
padding1_x, padding1_y = 500, 150
padding2_x, padding2_y = 300, 150
padding3_x, padding3_y = 300, 150

# -------------------- 머리 탐지 및 Homography 적용 함수 --------------------
def detect_heads_and_transform(image, H, padding_x, padding_y, color, window_name):
    results = model.predict(source=image, conf=0.3, save=False, verbose=False)[0]
    centers = []
    display_points = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centers.append([cx + padding_x, cy + padding_y])  # 패딩 보정
        display_points.append((cx + padding_x, cy + padding_y))
        cv2.circle(image, (int(cx), int(cy)), 5, color, -1)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    if centers:
        pts = np.array(centers, dtype=np.float32).reshape(-1, 1, 2)
        mapped_pts = cv2.perspectiveTransform(pts, H)
        return mapped_pts.reshape(-1, 2)
    else:
        return []

# -------------------- 탐지 및 변환 실행 --------------------
mapped1 = detect_heads_and_transform(img1, H1, padding1_x, padding1_y, (0, 0, 255), "Rear Image")
mapped2 = detect_heads_and_transform(img2, H2, padding2_x, padding2_y, (0, 255, 0), "Center Image")
mapped3 = detect_heads_and_transform(img3, H3, padding3_x, padding3_y, (255, 0, 0), "Front Image")

# -------------------- 좌석 정의 함수 --------------------
seat_width = 75
seat_height = 50
seat_start_x = 30
seat_start_y = 0

def generate_seat_boxes():
    boxes = []
    for row in range(2):
        num_seats = 7 if row == 0 else 5
        for col in range(num_seats):
            x = seat_start_x + col * seat_width
            y = seat_start_y + row * seat_height
            if row == 0 and col in [5, 6]:  # 좌석 6, 7 (index 기준 5, 6)
                y += 20  # y축 위로 20 띄우기
            boxes.append((x, y, seat_width, seat_height))

    rear_offsets = [2 * seat_width, 3 * seat_width + 10, 4 * seat_width + 20]  # 14번과 15번 사이에 추가 여백 10
    for offset in rear_offsets:
        x = seat_start_x + offset
        y = 190
        boxes.append((x, y, seat_width, seat_height))

    return boxes

# -------------------- 시각화 --------------------
plt.figure(figsize=(12, 6))
plt.xlim(0, 550)
plt.ylim(0, 240)
plt.axvspan(0, 370, facecolor='lightgray', alpha=0.2, label="front")
plt.axvspan(255, 425, facecolor='lightgreen', alpha=0.2, label="center")
plt.axvspan(350, 555, facecolor='lightblue', alpha=0.2, label="rear")

# 좌석 시각화
seat_boxes = generate_seat_boxes()
for i, (x, y, w, h) in enumerate(seat_boxes):
    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='orange', alpha=0.4, zorder=1)
    plt.gca().add_patch(rect)
    plt.text(x + w/2, y + h/2, f"S{i+1}", ha='center', va='center', fontsize=8, zorder=2)

# 머리 위치 시각화 다시 추가
if mapped1 is not None and len(mapped1) > 0:
    xs, ys = zip(*mapped1)
    plt.scatter(xs, ys, c='red', label="rear Head", s=60, zorder=3)

if mapped2 is not None and len(mapped2) > 0:
    xs, ys = zip(*mapped2)
    plt.scatter(xs, ys, c='green', label="center Head", s=60, zorder=3)

if mapped3 is not None and len(mapped3) > 0:
    xs, ys = zip(*mapped3)
    plt.scatter(xs, ys, c='blue', label="front Head", s=60, zorder=3)

plt.title("Bus seat coordinate (Real World)")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
