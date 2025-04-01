import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 📌 YOLO 모델 로드
model = YOLO("head.pt", verbose=False)

# 📌 데이터 폴더 경로 설정
side_folder = "output_scenario/scen1/camera6_image_raw_flat_single"
back_folder = "output_scenario/scen1/camera8_image_raw_flat_single"

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

# Homography 행렬 
H1 = np.array([
    [ 3.12293052e-01, -3.06614997e+00,  9.23450556e+02],
    [-5.61703036e-01, -5.89954372e-01,  5.55107180e+02],
    [ 6.31420942e-04, -4.62906929e-03 , 1.00000000e+00]
])

H2 = np.array([
    [ 6.06036999e-03,  2.15704280e-01, -2.04480699e+02],
    [ 4.31094911e-01, -3.99161955e-01, -1.56359721e+02],
    [ 1.00314085e-04, -2.97126407e-03,  1.00000000e+00]
])

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

# 공통 좌석 설정
seat_width = 75
seat_height = 50
seat_start_x = 30
seat_start_y = 0

# 좌석 좌표 계산 함수
def add_seat_boxes(ax):
    # 기본 좌석 (2줄)
    for row in range(2):
        num_seats = 7 if row == 0 else 5
        for col in range(num_seats):
            x = seat_start_x + col * seat_width
            y = seat_start_y + row * seat_height
            rect = plt.Rectangle((x, y), seat_width, seat_height,
                                 fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
            ax.add_patch(rect)
    # 추가 좌석 (3개)
    for offset in range(2, 5):
        x = seat_start_x + offset * seat_width
        y = 190
        rect = plt.Rectangle((x, y), seat_width, seat_height,
                             fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
        ax.add_patch(rect)

def render_rotated_plot(fig, temp_path, rotate_code):
    fig.savefig(temp_path, bbox_inches='tight')
    plt.close(fig)
    img = cv2.imread(temp_path)
    rotated = cv2.rotate(img, rotate_code)
    os.remove(temp_path)
    return rotated

for i in range(num_frames):
    img1 = side_images[i]
    img2 = back_images[i]
    
    padding1 = (300,150)  # 측면 패딩 보정값
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

    # 감지 및 변환
    people1, img1_viz = detect_people_in_real_area(img1, padding1, H1)
    people2, img2_viz = detect_people_in_real_area(img2, padding2, H2)

    transformed_pts1 = apply_homography(people1, H1)
    transformed_pts2 = apply_homography(people2, H2)
    
    if len(people1) < 1 or len(people2) < 1:
        print(f"❌ 검출된 사람이 부족함 (Frame {i})")
        continue

    # 후면 YOLO
    cv2.namedWindow("Back - YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Back - YOLO", 640, 640)
    cv2.imshow("Back - YOLO", img1_viz)
    cv2.moveWindow("Back - YOLO", 0, 0)

    # 후면 Homography
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(transformed_pts1[:, 0], transformed_pts1[:, 1], color='red', label="Cam1 (back)", alpha=0.6)
    ax1.set_title("Back Homography")
    ax1.grid(True)
    ax1.set_aspect('auto')
    add_seat_boxes(ax1)
    plt.tight_layout()
    rotated_back = render_rotated_plot(fig1, "temp_back.png", cv2.ROTATE_90_CLOCKWISE)

    cv2.namedWindow("Back Homography Rotated", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Back Homography Rotated", 600, 800)
    cv2.imshow("Back Homography Rotated", rotated_back)
    cv2.moveWindow("Back Homography Rotated", 700, 0)

    # 전면 YOLO
    cv2.namedWindow("Front - YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Front - YOLO", 640, 640)
    cv2.imshow("Front - YOLO", img2_viz)
    cv2.moveWindow("Front - YOLO", 1400, 0)

    # 전면 Homography
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.scatter(transformed_pts2[:, 0], transformed_pts2[:, 1], color='blue', label="Cam2 (front)", alpha=0.6)
    ax2.set_title("Front Homography")
    ax2.grid(True)
    ax2.set_aspect('auto')
    add_seat_boxes(ax2)
    plt.tight_layout()
    rotated_front = render_rotated_plot(fig2, "temp_front.png", cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.namedWindow("Front Homography Rotated", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Front Homography Rotated", 600, 800)
    cv2.imshow("Front Homography Rotated", rotated_front)
    cv2.moveWindow("Front Homography Rotated", 2100, 0)

    # waitKey & 종료
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == ord('q'):
        break

cv2.destroyAllWindows()
