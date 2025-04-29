import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def draw_seat_boxes_on_image(image, H_inv, padding, color=(0, 255, 255), seat_width=75, seat_height=50):
    seat_start_x = 30
    overlay = image.copy()

    # 📌 좌석 정의 (순서 중요: S1 ~ S15)
    seats = []

    # 앞줄 (좌석 1~7, 6~7은 y + 20)
    for col in range(7):
        x = seat_start_x + col * seat_width
        y = 0 if col < 5 else 20
        seats.append((x, y))

    # 뒷줄 (좌석 8~12)
    for col in range(5):
        x = seat_start_x + col * seat_width
        y = 50
        seats.append((x, y))

    # 후면 좌석 (13~15), 14~15 사이 여백 포함
    rear_offsets = [
        (2 * 75 , 170),               # 좌석 13
        (2 * 75 + 85, 170),          # 좌석 14 (y=170)
        (2 * 75 + 170, 170)           # 좌석 15 (y=170)
    ]
    for offset_x, y in rear_offsets:
        x = seat_start_x + offset_x
        seats.append((x, y))

    # 📐 좌석 박스를 투영하고 그리기
    for idx, (x, y) in enumerate(seats):
        if idx >= 12:    # S13, S14, S15
            seat_width, seat_height = 85, 70
        else:
            seat_width, seat_height = 75, 50

        box = np.array([[
            [x, y],
            [x + seat_width, y],
            [x + seat_width, y + seat_height],
            [x, y + seat_height]
        ]], dtype=np.float32)

        projected = cv2.perspectiveTransform(box, H_inv)[0]
        projected -= np.array(padding, dtype=np.float32)
        projected = projected.astype(int)
        cv2.polylines(overlay, [projected], isClosed=True, color=color, thickness=2)


    return overlay

# 📌 YOLO 모델 로드
model = YOLO("head.pt")

# 📌 이미지 로드
image1_path = "./frame_rear.jpg"  # Cam1 (측면)
image2_path = "./frame_center.jpg"  # Cam2-1 (후면)
imag3_path = "./frame_front.jpg"  # Cam2-2 (전면)

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)
img3 = cv2.imread(imag3_path)

if img1 is None or img2 is None:
    print("❌ 이미지 로드 실패")
    exit()

# 패딩 추가
padding1 = (500,150)
padding2 = (300,150)
padding3 = (300,150)

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

# 현실 좌표 영역 정의 (0,0)-(550,240)
REAL_AREA = np.array([[0, 0], [0, 240], [550, 240], [550, 0]], dtype=np.float32)

def inverse_homography(points, H):
    H_inv = np.linalg.inv(H)
    return cv2.perspectiveTransform(np.array([points], dtype=np.float32), H_inv)[0]

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0

def detect_people_in_real_area(image, padding, H):
    results = model(image)
    person_data = []
    img_copy = image.copy()
    H_inv = np.linalg.inv(H)

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                pt_image = np.array([[cx + padding[0], cy + padding[1]]], dtype=np.float32)
                pt_real = apply_homography(pt_image, H)
                if point_in_polygon(pt_real[0], REAL_AREA):
                    person_data.append(pt_image[0])
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
    return np.array(person_data, dtype=np.float32), img_copy

def apply_homography(points, H):
    return cv2.perspectiveTransform(np.array([points]), H)[0]

# 감지 및 변환
people1, img1_viz = detect_people_in_real_area(img1, padding1, H1)
people2, img2_viz = detect_people_in_real_area(img2, padding2, H2)
people3, img3_viz = detect_people_in_real_area(img3, padding3, H3)

transformed_pts1 = apply_homography(people1, H1)
transformed_pts2 = apply_homography(people2, H2)
transformed_pts3 = apply_homography(people3, H3)

def visualize_results(img1_viz, img2_viz, img3_viz, pts1, pts2, pts3):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img1_viz, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image 1 - YOLO Detected (rear)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2 - YOLO Detected (center)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(img3_viz, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Image 3 - YOLO Detected (front)")
    axes[2].axis("off")

    axes[3].scatter(pts1[:, 0], pts1[:, 1], color='red', label="Transformed Cam (rear)", alpha=0.6)
    axes[3].scatter(pts2[:, 0], pts2[:, 1], color='blue', label="Transformed Cam (center)", alpha=0.6)
    axes[3].scatter(pts3[:, 0], pts3[:, 1], color='green', label="Transformed Cam (front)", alpha=0.6)
    axes[3].set_xlabel("X-axis")
    axes[3].set_ylabel("Y-axis")
    axes[3].set_title("Homography 2D Coordinates")

    # 좌석 시각화
    seat_width = 75
    seat_height = 50
    seat_start_x = 30

    # 좌석 좌표 리스트 정의
    seat_positions = []

    # 앞줄 (좌석 1~7, 6~7은 y + 20)
    for col in range(7):
        x = seat_start_x + col * seat_width
        y = 0 if col < 5 else 20
        seat_positions.append((x, y))

    # 뒷줄 (좌석 8~12)
    for col in range(5):
        x = seat_start_x + col * seat_width
        y = 50
        seat_positions.append((x, y))

    # 후면 좌석 (13~15), 14~15는 y=170
    rear_offsets = [2 * seat_width, 3 * seat_width + 10, 4 * seat_width + 20]
    rear_ys = [190, 170, 170]
    for offset, y in zip(rear_offsets, rear_ys):
        x = seat_start_x + offset
        seat_positions.append((x, y))

    # 시각화
    for (x, y) in seat_positions:
        rect = plt.Rectangle((x, y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
        axes[3].add_patch(rect)

    axes[3].legend()
    axes[3].grid(True)
    plt.tight_layout()
    plt.show()

visualize_results(img1_viz, img2_viz, img3_viz ,transformed_pts1, transformed_pts2, transformed_pts3)
# 역행렬 계산
H1_inv = np.linalg.inv(H1)
H2_inv = np.linalg.inv(H2)
H3_inv = np.linalg.inv(H3)

# 이미지에 좌석 박스 그리기 (패딩 보정 포함)
img1_with_seats = draw_seat_boxes_on_image(img1.copy(), H1_inv, padding1)
img2_with_seats = draw_seat_boxes_on_image(img2.copy(), H2_inv, padding2)
img3_with_seats = draw_seat_boxes_on_image(img3.copy(), H3_inv, padding3)

cv2.imshow("Cam1 with Seat Boxes", img1_with_seats)
cv2.imshow("Cam2 with Seat Boxes", img2_with_seats)
cv2.imshow("Cam3 with Seat Boxes", img3_with_seats)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
