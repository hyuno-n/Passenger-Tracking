import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === YOLO 모델 로드
model = YOLO("head.pt")

# === 좌석 2개 단위의 실좌표 (좌석 평면)
seat_real_coords = [
    np.array([[30, 0], [30, 100], [105, 100], [105, 0]], dtype=np.float32),   # 좌석 1+2
    np.array([[105, 0], [105, 100], [180, 100], [180, 0]], dtype=np.float32), # 좌석 3+4
    np.array([[180, 0], [180, 100], [255, 100], [255, 0]], dtype=np.float32), # 좌석 5+6
]

# === 이미지 내 좌석 대응 좌표
seat_image_coords = [
    np.array([(537, 155), (639, 254), (428, 255), (393, 158)], dtype=np.float32),
    np.array([(393, 158), (427, 256), (214, 264), (257, 163)], dtype=np.float32),
    np.array([(257, 163), (216, 265), (39, 275), (134, 171)], dtype=np.float32),
]

# === 이미지 로드
image = cv2.imread("image1.jpg")
if image is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# === Homography 및 역행렬 계산
homographies = []
homographies_inv = []
for i in range(3):
    H, _ = cv2.findHomography(seat_image_coords[i], seat_real_coords[i])
    homographies.append(H)
    homographies_inv.append(np.linalg.inv(H))

# === 머리 중심점 검출
def detect_heads(image):
    results = model(image)
    head_points = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # head class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                head_points.append([cx, cy])
    return np.array(head_points, dtype=np.float32)

head_points = detect_heads(image)

# === 변환 함수
def transform_point(pt, H):
    pt = np.array([[pt]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

# === 점이 사각형 안에 있는지 확인
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0

# === 2좌석을 나눠서 각각 좌석 박스로 투영
def draw_split_seats(image, seat_real_rect, H_inv, base_seat_idx, color=(0, 255, 255)):
    x1, _ = seat_real_rect[0]
    x2, _ = seat_real_rect[2]

    # bottom (좌석 a)
    seat_bottom = np.array([
        [x1, 0],
        [x2, 0],
        [x2, 50],
        [x1, 50]
    ], dtype=np.float32)

    # top (좌석 b)
    seat_top = np.array([
        [x1, 50],
        [x2, 50],
        [x2, 100],
        [x1, 100]
    ], dtype=np.float32)

    for i, seat in enumerate([seat_bottom, seat_top]):
        projected = cv2.perspectiveTransform(np.array([seat]), H_inv)[0]
        projected = projected.astype(int)
        cv2.polylines(image, [projected], isClosed=True, color=color, thickness=2)
        # 번호 표시
        cx, cy = projected.mean(axis=0).astype(int)
        cv2.putText(image, f"{base_seat_idx+i}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# === matplotlib plot에서도 split 좌석 그리기
def draw_split_seats_on_plot(ax, seat_real_rect, base_seat_idx, color='gray'):
    x1, _ = seat_real_rect[0]
    x2, _ = seat_real_rect[2]

    for i, y_start in enumerate([0, 50]):
        seat_rect = [
            (x1, y_start),
            (x2, y_start),
            (x2, y_start + 50),
            (x1, y_start + 50)
        ]
        ax.plot(*zip(*seat_rect, seat_rect[0]), color=color, linestyle='--')
        ax.fill(*zip(*seat_rect), alpha=0.1)
        cx = (x1 + x2) / 2
        cy = y_start + 25
        ax.text(cx, cy, f"{base_seat_idx + i}", ha='center', va='center', fontsize=9, color='black')

# === 이미지에 좌석 박스 투영
image_with_boxes = image.copy()
seat_index = 1
for i in range(3):
    draw_split_seats(image_with_boxes, seat_real_coords[i], homographies_inv[i], base_seat_idx=seat_index)
    seat_index += 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 두 개 subplot

colors = ['red', 'green', 'blue']
seat_index = 1

# === 좌석만 표시: ax1
for i in range(3):
    draw_split_seats_on_plot(ax1, seat_real_coords[i], base_seat_idx=seat_index)
    seat_index += 2
ax1.set_xlim(0, 300)
ax1.set_ylim(0, 120)
ax1.invert_xaxis()
ax1.invert_yaxis()
ax1.set_title("Seat Layout")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid(True)

# === 좌석 + 머리점 표시: ax2
seat_index = 1
for i in range(3):
    draw_split_seats_on_plot(ax2, seat_real_coords[i], base_seat_idx=seat_index)
    seat_index += 2

for pt in head_points:
    for i, seat_poly in enumerate(seat_image_coords):
        if point_in_polygon(pt, seat_poly):
            transformed = transform_point(pt, homographies[i])
            x1, _ = seat_real_coords[i][0]
            x2, _ = seat_real_coords[i][2]
            seat_bottom = np.array([[x1, 0], [x2, 0], [x2, 50], [x1, 50]], dtype=np.float32)
            seat_top    = np.array([[x1, 50], [x2, 50], [x2, 100], [x1, 100]], dtype=np.float32)
            cx, cy = map(int, pt)
            cv2.circle(image_with_boxes, (cx, cy), 3, (0, 0, 255), -1)  # 빨간 점
            if point_in_polygon(transformed, seat_bottom):
                print(f"🪑 머리점 {pt} → 좌석 {2*i+1} (bottom)")
            elif point_in_polygon(transformed, seat_top):
                print(f"🪑 머리점 {pt} → 좌석 {2*i+2} (top)")

            ax2.scatter(*transformed, color=colors[i], marker='x')
            ax2.text(transformed[0], transformed[1] - 3, f"H{i}", fontsize=8, color=colors[i])
            break

ax2.set_xlim(0, 300)
ax2.set_ylim(0, 120)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.set_title("Head Detection")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True)

plt.tight_layout()
plt.show()

# === OpenCV로 이미지 시각화
cv2.imshow("Seat Projection on Image", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
