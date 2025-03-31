import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def draw_seat_boxes_on_image(image, H_inv, padding, color=(0, 255, 255), seat_width=75, seat_height=50):
    seat_start_x = 30
    seat_start_y = 0
    overlay = image.copy()

    def project_and_draw_box(seat_box):
        projected = cv2.perspectiveTransform(np.array([seat_box], dtype=np.float32), H_inv)[0]
        # 패딩 보정
        projected -= np.array(padding, dtype=np.float32)
        projected = projected.astype(int)
        cv2.polylines(overlay, [projected], isClosed=True, color=color, thickness=2)

    # 🔹 아래줄: 7개
    for col in range(4):
        x = seat_start_x + col * seat_width
        y = seat_start_y + seat_height  # 아래 줄 (row 1)
        box = [
            [x, y],
            [x + seat_width, y],
            [x + seat_width, y + seat_height],
            [x, y + seat_height]
        ]
        project_and_draw_box(box)

    # 🔹 위줄: 5개
    for col in range(4):
        x = seat_start_x + col * seat_width
        y = seat_start_y  # 위 줄 (row 0)
        box = [
            [x, y],
            [x + seat_width, y],
            [x + seat_width, y + seat_height],
            [x, y + seat_height]
        ]
        project_and_draw_box(box)

    # 🔹 추가 좌석: y=190
    extra_y = 190
    for col in [2, 3, 3]:  # 3번 좌석이 중복되었지만 그대로 유지
        x = seat_start_x + col * seat_width
        box = [
            [x, extra_y],
            [x + seat_width, extra_y],
            [x + seat_width, extra_y + seat_height],
            [x, extra_y + seat_height]
        ]
        project_and_draw_box(box)

    return overlay

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
padding1 = (700,500)
padding2 = (300,150)

# 📌 Homography 행렬
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

# 현실 좌표 영역 정의 (0,0)-(550,240)
REAL_AREA = np.array([[0, 0], [0, 240], [330, 240], [330, 0]], dtype=np.float32)

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

transformed_pts1 = apply_homography(people1, H1)
transformed_pts2 = apply_homography(people2, H2)

def visualize_results(img1_viz, img2_viz, pts1, pts2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img1_viz, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image 1 - YOLO Detected (side)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2 - YOLO Detected (back)")
    axes[1].axis("off")

    axes[2].scatter(pts1[:, 0], pts1[:, 1], color='red', label="Transformed Cam1 (side)", alpha=0.6)
    axes[2].scatter(pts2[:, 0], pts2[:, 1], color='blue', label="Transformed Cam2-1 (back)", alpha=0.6)
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")
    axes[2].set_title("Homography 2D Coordinates")

    # 좌석 시각화
    seat_width = 75
    seat_height = 50
    seat_start_x = 30
    seat_start_y = 0
    for row in range(2):  # 총 2줄
        if row == 1:
            num_seats = 4  # 위 줄에 4개
        else:
            num_seats = 4  # 아래 줄에 6개
        for col in range(num_seats):
            x = seat_start_x + col * seat_width
            y = seat_start_y + row * seat_height
            rect = plt.Rectangle((x, y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
            axes[2].add_patch(rect)

    extra_seat_y = 190
    extra_seat_x_1 = seat_start_x + 2 * seat_width
    extra_seat_x_2 = seat_start_x + 3 * seat_width
    # extra_seat_x_3 = seat_start_x + 3 * seat_width
    rect1 = plt.Rectangle((extra_seat_x_1, extra_seat_y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
    rect2 = plt.Rectangle((extra_seat_x_2, extra_seat_y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
    # rect3 = plt.Rectangle((extra_seat_x_3, extra_seat_y), seat_width, seat_height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
    axes[2].add_patch(rect1)
    axes[2].add_patch(rect2)
    # axes[2].add_patch(rect3)

    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()

# visualize_results(img1_viz, img2_viz, transformed_pts1, transformed_pts2)
# 역행렬 계산
H1_inv = np.linalg.inv(H1)
H2_inv = np.linalg.inv(H2)

# 이미지에 좌석 박스 그리기 (패딩 보정 포함)
img1_with_seats = draw_seat_boxes_on_image(img1.copy(), H1_inv, padding1)
img2_with_seats = draw_seat_boxes_on_image(img2.copy(), H2_inv, padding2)

cv2.imshow("Cam1 with Seat Boxes", img1_with_seats)
cv2.imshow("Cam2 with Seat Boxes", img2_with_seats)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
