# Homography 기반 착석 인식 + 시각화 + CSV 저장 전체 코드

import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("head.pt", verbose=False)

# 디렉토리 설정
base_dir = "output_scenario"
scenario_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("scen")])
camera_dirs = ["camera6_image_raw_flat_single", "camera8_image_raw_flat_single"]

# Homography 행렬 및 패딩
H_map = {
    "camera6_image_raw_flat_single": np.array([
        [3.12293052e-01, -3.06614997e+00,  9.23450556e+02],
        [-5.61703036e-01, -5.89954372e-01,  5.55107180e+02],
        [6.31420942e-04, -4.62906929e-03,  1.00000000e+00]
    ]),
    "camera8_image_raw_flat_single": np.array([
        [6.06036999e-03,  2.15704280e-01, -2.04480699e+02],
        [4.31094911e-01, -3.99161955e-01, -1.56359721e+02],
        [1.00314085e-04, -2.97126407e-03,  1.00000000e+00]
    ])
}
padding = (300, 150)
REAL_AREA = np.array([[0, 0], [0, 240], [550, 240], [550, 0]], dtype=np.float32)

# 좌석 정의
seat_width, seat_height = 75, 50
seat_start_x = 30
seats = []
for row in range(2):
    num_seats = 7 if row == 0 else 5
    for col in range(num_seats):
        x = seat_start_x + col * seat_width
        y = row * seat_height
        seats.append((x, y))
seats += [(seat_start_x + i * seat_width, 190) for i in range(2, 5)]

# 유틸 함수
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def load_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")], key=extract_number)
    return [cv2.imread(os.path.join(folder, f)) for f in files if cv2.imread(os.path.join(folder, f)) is not None], files

def apply_homography(points, H):
    return cv2.perspectiveTransform(np.array([points], dtype=np.float32), H)[0]

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0

def detect_people_in_real_area(image, padding, H):
    results = model(image)
    person_data = []
    img_copy = image.copy()
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2 + padding[0]
                cy = (y1 + y2) // 2 + padding[1]
                pt_img = np.array([[cx, cy]], dtype=np.float32)
                pt_real = apply_homography(pt_img, H)
                if point_in_polygon(pt_real[0], REAL_AREA):
                    person_data.append(pt_img[0])
    return np.array(person_data, dtype=np.float32)

# 결과 저장용 리스트
records = []

for scen in scenario_dirs:
    cam1_path = os.path.join(base_dir, scen, camera_dirs[0])
    cam2_path = os.path.join(base_dir, scen, camera_dirs[1])
    cam1_imgs, cam1_files = load_images(cam1_path)
    cam2_imgs, cam2_files = load_images(cam2_path)
    num_frames = min(len(cam1_imgs), len(cam2_imgs))

    for i in range(num_frames):
        img1, img2 = cam1_imgs[i], cam2_imgs[i]
        fname = cam1_files[i]

        H1, H2 = H_map[camera_dirs[0]], H_map[camera_dirs[1]]
        people1 = detect_people_in_real_area(img1, padding, H1)
        people2 = detect_people_in_real_area(img2, padding, H2)

        if len(people1) == 0 and len(people2) == 0:
            continue

        trans1 = apply_homography(people1, H1)
        trans2 = apply_homography(people2, H2)

        # 좌석 점유 여부 판단
        occupied = [0] * len(seats)
        for (px, py) in np.concatenate((trans1, trans2)):
            for idx, (x, y) in enumerate(seats):
                if x <= px <= x + seat_width and y <= py <= y + seat_height:
                    occupied[idx] = 1

        record = {"scenario": scen, "frame": fname}
        for idx, val in enumerate(occupied):
            record[f"seat_{idx}"] = val
        records.append(record)

# CSV 저장
pd.DataFrame(records).to_csv("homography_occupancy_result.csv", index=False)
print("✅ homography_occupancy_result.csv 저장 완료")
