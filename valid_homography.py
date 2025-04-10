import os
import json
import re
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# YOLO 모델 로드
model = YOLO("head.pt")

# 📌 좌석 정의 (순서 중요: S1 ~ S15)
seat_width, seat_height = 75, 50
seat_start_x = 30
seats = []
for row in range(2):
    num_seats = 7 if row == 0 else 5
    for col in range(num_seats):
        x = seat_start_x + col * seat_width
        y = row * seat_height
        seats.append((x, y))
seats += [(seat_start_x + i * seat_width, 190) for i in range(2, 5)]  # 13~15

# Homography 행렬 + padding (각도별)
homographies = {
    "view_40": (np.array([[ 3.39218386e-01, -2.61378020e+00, -2.41384154e+02],
                          [ 1.23077482e+00, -1.09011484e+00, -8.55740148e+02],
                          [ 8.33939613e-04, -9.20352638e-03,  1.00000000e+00]]), (500, 150)),
    "view_0":  (np.array([[-2.81607724e-01,  1.04586005e-01,  5.69682600e+02],
                          [ 5.25878026e-02,  3.67465386e-01, -8.36607016e+01],
                          [ 2.01001615e-04,  2.72166839e-04,  1.00000000e+00]]), (300, 150)),
    "view_-40":(np.array([[-6.29676468e-01, -5.20759458e+00,  1.44638387e+03],
                          [-1.89307888e+00, -1.34889107e+00,  1.32959556e+03],
                          [-1.56119349e-03, -1.07253880e-02,  1.00000000e+00]]), (300, 150))
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

def plot_seat_confusion(seat_stats):
    metrics = ["TP", "FP", "FN", "TN"]
    seat_labels = [f"S{i}" for i in range(1, 16)]

    data = {m: [seat_stats[s][m] for s in seat_labels] for m in metrics}

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for i, metric in enumerate(metrics):
        sns.heatmap(
            np.array(data[metric]).reshape(1, -1),
            annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=seat_labels, yticklabels=[metric],
            ax=axes[i], cbar=False
        )
        axes[i].set_title(f"{metric} per seat")
    plt.tight_layout()
    plt.show()

# ========= 평가 루프 =========
base_dir = "data/scen_output"
label_dir = "data/scen_label"

total = defaultdict(int)

for scen in sorted(os.listdir(base_dir)):
    if not scen.startswith("scen"):
        continue

    label_path = os.path.join(label_dir, f"{scen}.json")
    if not os.path.isfile(label_path):
        print(f"⚠️ 라벨 없음: {label_path}")
        continue

    with open(label_path, "r") as f:
        label_data = json.load(f)

    # 각 뷰 경로
    view_paths = {
        view: os.path.join(base_dir, scen, view)
        for view in ["view_-40", "view_0", "view_40"]
    }

    # 파일 목록
    filenames = sorted([f for f in os.listdir(view_paths["view_0"]) if f.endswith(".jpg")], key=extract_number)
    seat_stats = {sid: {"TP":0, "FP":0, "FN":0, "TN":0} for sid in seat_ids}
    for fname in tqdm(filenames, desc=f"🎞 {scen}"):
        pred_people = []
        for view, path in view_paths.items():
            img_path = os.path.join(path, fname)
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            H, pad = homographies[view]
            pred_people += detect_heads(img, H, pad)

        pred_vector = predict_occupancy(pred_people)

        # Ground Truth
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
        plot_seat_confusion(seat_stats)

# ========= 결과 출력 =========
print("\n📊 좌석 점유 평가 결과")
print(f"True Positive (TP): {total['TP']}")
print(f"False Positive (FP): {total['FP']}")
print(f"False Negative (FN): {total['FN']}")
print(f"True Negative (TN): {total['TN']}")
