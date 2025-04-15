import os
import json
import re
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

# YOLO 모델 로드
model = YOLO("runs/detect/head_finetuned2/weights/best.pt")

# 📌 좌석 정의 (순서 중요: S1 ~ S15)
seat_width, seat_height = 75, 50
seat_start_x = 30
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
rear_offsets = [2 * seat_width, 3 * seat_width + 10, 4 * seat_width + 20]
for offset in rear_offsets:
    x = seat_start_x + offset
    y = 190
    seats.append((x, y))

# Homography 행렬 + padding (각도별)
homographies_default = {
    "view_40": (np.array([[ 3.39218386e-01, -2.61378020e+00, -2.41384154e+02],
                           [ 1.23077482e+00, -1.09011484e+00, -8.55740148e+02],
                           [ 8.33939613e-04, -9.20352638e-03,  1.00000000e+00]]), (500, 150)),
    "view_0": (np.array([[ -1.84509850e-01,  8.03468203e-02,  5.25063189e+02],
                          [ 4.81525443e-02,  3.72219168e-01, -8.28806408e+01],
                          [ 2.24470429e-04,  2.05735101e-04,  1.00000000e+00]]), (300, 150)),
    "view_-40": (np.array([[ -5.95639903e-01, -4.92610298e+00,  1.36820095e+03],
                            [-1.89307888e+00, -1.34889107e+00,  1.32959556e+03],
                            [-1.56119349e-03, -1.07253880e-02,  1.00000000e+00]]), (300, 150)),
}

homographies_alt = {
    "view_40": (np.array([[-1.46210375e+00,  8.60075220e+00,  2.73692268e+03],
                           [-5.29493124e+00,  4.18599281e+00,  4.20388569e+03],
                           [-4.21451893e-03,  3.45669963e-02,  1.00000000e+00]]), (500, 150)),
    "view_0": (np.array([[-2.54814779e-01,  3.02230300e-02,  4.89775051e+02],
                          [-5.03690736e-04,  3.02718132e-01, -4.52354576e+01],
                          [-1.52079224e-05,  7.02600897e-05,  1.00000000e+00]]), (300, 150)),
    "view_-40": (np.array( [[-2.42508598e-01, -2.06890148e+00,  7.57801477e+02],
                            [-7.87968226e-01, -5.71032036e-01,  6.23032341e+02],
                            [-6.41045657e-04, -4.78356023e-03,  1.00000000e+00]]), (300, 150)),
}

REAL_AREA = np.array([[0, 0], [0, 240], [550, 240], [550, 0]], dtype=np.float32)
seat_ids = [f"S{i}" for i in range(1, 16)]

# ========= 유틸 함수 =========
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def apply_homography(points, H):
    return cv2.perspectiveTransform(np.array([points], dtype=np.float32), H)[0]

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0

def detect_heads(image, H, padding):
    results = model(image, verbose=False)
    people = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2 + padding[0]
                cy = (y1 + y2) // 2 + padding[1]
                pt = np.array([[cx, cy]], dtype=np.float32)
                real = apply_homography(pt, H)
                if point_in_polygon(real[0], REAL_AREA):
                    people.append(real[0])
    return people

def predict_occupancy(people):
    occupied = [0] * len(seats)
    for (px, py) in people:
        for idx, (x, y) in enumerate(seats):
            if x <= px <= x + seat_width and y <= py <= y + seat_height:
                occupied[idx] = 1
    return occupied

def visualize_seat_detection_rate(seat_stats):
    # 전체 TP/FP/FN/TN 출력
    total = defaultdict(int)
    for sid in seat_ids:
        for k in seat_stats[sid]:
            total[k] += seat_stats[sid][k]

    print("📊 Overall Evaluation Results")
    print(f"True Positive (TP): {total['TP']}")
    print(f"False Positive (FP): {total['FP']}")
    print(f"False Negative (FN): {total['FN']}")
    print(f"True Negative (TN): {total['TN']}")

    seat_labels = [f"S{i}" for i in range(1, 16)]
    acc = []
    for s in seat_labels:
        tp = seat_stats[s]["TP"]
        fn = seat_stats[s]["FN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc.append(recall)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 240)
    ax.invert_yaxis()
    ax.set_title("Seat-wise Detection Recall")
    for i, (x, y) in enumerate(seats):
        color = plt.cm.Reds(acc[i])  # 강도에 따라 빨간색 음영
        rect = plt.Rectangle((x, y), seat_width, seat_height, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + seat_width/2, y + seat_height/2, f"{seat_labels[i]}\n{acc[i]*100:.1f}%",
                ha='center', va='center', fontsize=9, color='white' if acc[i] > 0.5 else 'black')
    # plt.tight_layout()
    # plt.show()

# ========= 평가 루프 =========
base_dir = "data/scen_output"
label_dir = "data/scen_label"

for scen in sorted(os.listdir(base_dir), key=lambda s: int(re.search(r'scen(\d+)', s).group(1))):
    if not scen.startswith("scen"):
        continue

    label_path = os.path.join(label_dir, f"{scen}.json")
    if not os.path.isfile(label_path):
        print(f"⚠️ 라벨 없음: {label_path}")
        continue

    with open(label_path, "r") as f:
        label_data = json.load(f)

    view_paths = {
        view: os.path.join(base_dir, scen, view)
        for view in ["view_-40", "view_0", "view_40"]
    }

    filenames = sorted([f for f in os.listdir(view_paths["view_0"]) if f.endswith(".jpg")], key=extract_number)
    seat_stats = {sid: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for sid in seat_ids}

    for fname in tqdm(filenames, desc=f"🎞 {scen}"):
        pred_people = []
        for view, path in view_paths.items():
            img_path = os.path.join(path, fname)
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            scenario_index = int(re.search(r'scen(\d+)', scen).group(1))
            H, pad = (homographies_alt if scenario_index >= 7 else homographies_default)[view]
            pred_people += detect_heads(img, H, pad)

        pred_vector = predict_occupancy(pred_people)

        if fname not in label_data:
            continue
        gt_dict = label_data[fname]
        gt_vector = [int(gt_dict.get(sid, False)) for sid in seat_ids]

        for p, g, sid in zip(pred_vector, gt_vector, seat_ids):
            if g == 1 and p == 1:
                seat_stats[sid]["TP"] += 1
            elif g == 1 and p == 0:
                seat_stats[sid]["FN"] += 1
            elif g == 0 and p == 1:
                seat_stats[sid]["FP"] += 1
            else:
                seat_stats[sid]["TN"] += 1

    visualize_seat_detection_rate(seat_stats)
