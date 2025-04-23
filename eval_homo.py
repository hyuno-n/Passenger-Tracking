import os
import json
import re
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO

# ✅ 모델 로딩
model = YOLO("head.pt")

# ✅ Homography 행렬 및 패딩 설정
homographies = {
    "front": {
        "H": np.array([
            [6.06036999e-03,  2.15704280e-01, -2.04480699e+02],
            [4.31094911e-01, -3.99161955e-01, -1.56359721e+02],
            [1.00314085e-04, -2.97126407e-03, 1.00000000e+00]
        ]),
        "pad": (300, 150)
    },
    "rear": {
        "H": np.array([
            [3.12293052e-01, -3.06614997e+00, 9.23450556e+02],
            [-5.61703036e-01, -5.89954372e-01, 5.55107180e+02],
            [6.31420942e-04, -4.62906929e-03, 1.00000000e+00]
        ]),
        "pad": (300, 150)
    }
}

# ✅ 좌석 정보
def define_seats():
    seat_width = 75
    seat_start_x = 30
    seats = []

    # 앞줄 (S1~S7), 6~7번은 y = 20
    for col in range(7):
        x = seat_start_x + col * seat_width
        y = 0 if col < 5 else 20
        seats.append((x, y))

    # 뒷줄 (S8~S12)
    for col in range(5):
        x = seat_start_x + col * seat_width
        y = 50
        seats.append((x, y))

    # 후면 좌석 (S13~S15), 여백 포함한 위치 조정
    rear_xs = [
        seat_start_x + 2 * seat_width,
        seat_start_x + 2 * seat_width + 85,
        seat_start_x + 2 * seat_width + 170
    ]
    rear_y = 170
    for x in rear_xs:
        seats.append((x, rear_y))

    return seats

SEATS = define_seats()
REAL_AREA = np.array([[0, 0], [0, 240], [550, 240], [550, 0]], dtype=np.float32)
SEAT_IDS = [f"S{i}" for i in range(1, 16)]

# ✅ 유틸 함수
def get_seat_size(index):
    if index >= 12:  # S13~S15
        return 85, 70
    else:
        return 75, 50

def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else float("inf")

def apply_homography(points, H):
    return cv2.perspectiveTransform(np.array([points], dtype=np.float32), H)[0]

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0

def detect_heads(image, H, padding):
    results = model(image, conf=0.5, verbose=False)
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
    occupied = [0] * len(SEATS)
    for (px, py) in people:
        for idx, (x, y) in enumerate(SEATS):
            w, h = get_seat_size(idx)
            if x <= px <= x + w and y <= py <= y + h:
                occupied[idx] = 1
    return occupied

# ✅ 평가 루프
base_dir = "data/front_rear_data"
label_dir = "data/scen_label/front_rear_camera"

def evaluate_all_scenarios():
    for scen in sorted(os.listdir(base_dir), key=lambda s: int(re.search(r"\d+", s).group(0))):
        scen_path = os.path.join(base_dir, scen)
        label_path = os.path.join(label_dir, f"{scen}.json")
        if not os.path.isfile(label_path):
            print(f"⚠️ 라벨 없음: {label_path}")
            continue

        with open(label_path, "r") as f:
            gt_data = json.load(f)

        filenames = sorted(os.listdir(os.path.join(scen_path, "front")), key=extract_number)
        stats = {sid: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for sid in SEAT_IDS}

        for fname in tqdm(filenames, desc=f"🎞 {scen}"):
            all_points = []
            for cam in ["front", "rear"]:
                img_path = os.path.join(scen_path, cam, fname)
                if not os.path.exists(img_path): continue
                img = cv2.imread(img_path)
                H = homographies[cam]["H"]
                pad = homographies[cam]["pad"]
                all_points += detect_heads(img, H, pad)

            pred = predict_occupancy(all_points)
            if fname not in gt_data:
                continue
            gt = [int(gt_data[fname].get(sid, False)) for sid in SEAT_IDS]

            for p, g, sid in zip(pred, gt, SEAT_IDS):
                if p == 1 and g == 1:
                    stats[sid]["TP"] += 1
                elif p == 1 and g == 0:
                    stats[sid]["FP"] += 1
                elif p == 0 and g == 1:
                    stats[sid]["FN"] += 1
                else:
                    stats[sid]["TN"] += 1

        print(f"\n📊 {scen} 결과:")
        TP = sum([s["TP"] for s in stats.values()])
        FP = sum([s["FP"] for s in stats.values()])
        FN = sum([s["FN"] for s in stats.values()])
        TN = sum([s["TN"] for s in stats.values()])
        print(f"TP: {TP} | FP: {FP} | FN: {FN} | TN: {TN}")
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1 Score: {f1:.3f}")

if __name__ == "__main__":
    evaluate_all_scenarios()
