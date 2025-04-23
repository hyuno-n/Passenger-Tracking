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
model = YOLO("runs/detect/finetune_freeze/weights/best.pt")

seat_width = 75
seat_height = 50
seat_start_x = 30
seats = []

# 앞줄 (S1~S7), S6~S7 (index 5,6)는 y + 20
for col in range(7):
    x = seat_start_x + col * seat_width
    y = 0 if col < 5 else 20
    seats.append((x, y))

# 뒷줄 (S8~S12)
for col in range(5):
    x = seat_start_x + col * seat_width
    y = 50
    seats.append((x, y))

# 후면 좌석 (S13~S15), 여백 반영
rear_xs = [
    seat_start_x + 2 * seat_width,
    seat_start_x + 2 * seat_width + 85,
    seat_start_x + 2 * seat_width + 170
]
rear_y = 170
for x in rear_xs:
    seats.append((x, rear_y))

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
def get_seat_size(index):
    if index >= 12:  # S13~S15
        return 85, 70
    else:
        return 75, 50

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

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
    occupied = [0] * len(seats)
    for (px, py) in people:
        for idx, (x, y) in enumerate(seats):
            w, h = get_seat_size(idx)
            if x <= px <= x + w and y <= py <= y + h:
                occupied[idx] = 1
    return occupied

def visualize_seat_detection_rate(seat_stats):
    from collections import defaultdict
    total = defaultdict(int)
    for sid in seat_ids:
        for k in seat_stats[sid]:
            total[k] += seat_stats[sid][k]
    print("📊 Overall Evaluation Results")
    print(f"True Positive (TP): {total['TP']}")
    print(f"False Positive (FP): {total['FP']}")
    print(f"False Negative (FN): {total['FN']}")
    print(f"True Negative (TN): {total['TN']}")

    acc = []
    for sid in seat_ids:
        tp = seat_stats[sid]["TP"]
        fn = seat_stats[sid]["FN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc.append(recall)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 240)
    ax.invert_yaxis()
    ax.set_title("Seat-wise Detection Recall")

    for i, (x, y) in enumerate(seats):
        w, h = get_seat_size(i)
        color = plt.cm.Reds(acc[i])
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, f"{seat_ids[i]}\n{acc[i]*100:.1f}%",
                ha='center', va='center', fontsize=9, color='white' if acc[i] > 0.5 else 'black')
    plt.tight_layout()
    plt.show()

def draw_predictions(img_path, view, scenario_index):
    img = cv2.imread(img_path)
    img_height = img.shape[0]
    H, pad = (homographies_alt if scenario_index >= 7 else homographies_default)[view]
    results = model(img, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # 바운딩 박스
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 텍스트 위치: 기본적으로 박스 위, 위로 벗어나면 박스 아래
                text = f"{conf:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_y = y1 - 10 if y1 - 10 > text_size[1] else y2 + 20
                text_y = min(text_y, img_height - 5)  # 아래로 벗어나는 것도 방지
                cv2.putText(img, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



def show_fp_images(fp_frames, scenario_index, sample_per_view=3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, view in enumerate(["view_-40", "view_0", "view_40"]):
        paths = fp_frames[view][:sample_per_view]
        combined = None
        for path in paths:
            img = draw_predictions(path, view, scenario_index)
            combined = img if combined is None else cv2.vconcat([combined, img])
        if combined is not None:
            axes[idx].imshow(combined)
        axes[idx].set_title(f"{view} (FP with detection)", fontsize=14)
        axes[idx].axis("off")
    plt.tight_layout()
    plt.show()

def show_fn_images(fn_frames, scenario_index, sample_per_view=3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, view in enumerate(["view_-40", "view_0", "view_40"]):
        paths = fn_frames[view][:sample_per_view]
        combined = None
        for path in paths:
            img = draw_predictions(path, view, scenario_index)
            combined = img if combined is None else cv2.vconcat([combined, img])
        if combined is not None:
            axes[idx].imshow(combined)
        axes[idx].set_title(f"{view} (FN with detection)", fontsize=14)
        axes[idx].axis("off")
    plt.tight_layout()
    plt.show()

# ========= conf threshold별 성능 평가 루프 =========
def evaluate_conf_range(model, base_dir, label_dir, seat_ids, apply_homography, point_in_polygon, predict_occupancy, extract_number, homographies_default, homographies_alt, REAL_AREA):
    from matplotlib import pyplot as plt
    import os
    import cv2
    import json
    import numpy as np
    import re
    from tqdm import tqdm

    conf_range = [round(x * 0.1, 2) for x in range(1, 10)]
    conf_results = []

    for conf_thres in conf_range:
        total_stats = {sid: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for sid in seat_ids}
        for scen in sorted(os.listdir(base_dir), key=lambda s: int(re.search(r'scen(\d+)', s).group(1))):
            if not scen.startswith("scen"):
                continue
            label_path = os.path.join(label_dir, f"{scen}.json")
            if not os.path.isfile(label_path):
                continue
            with open(label_path, "r") as f:
                label_data = json.load(f)
            view_paths = {
                view: os.path.join(base_dir, scen, view)
                for view in ["view_-40", "view_0", "view_40"]
            }
            filenames = sorted([f for f in os.listdir(view_paths["view_0"]) if f.endswith(".jpg")], key=extract_number)
            for fname in tqdm(filenames, desc=f"🎞 {scen} (conf={conf_thres})"):
                pred_people = []
                for view, path in view_paths.items():
                    img_path = os.path.join(path, fname)
                    if not os.path.isfile(img_path):
                        continue
                    img = cv2.imread(img_path)
                    scenario_index = int(re.search(r'scen(\d+)', scen).group(1))
                    H, pad = (homographies_alt if scenario_index >= 7 else homographies_default)[view]
                    results = model(img, conf=conf_thres, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            if int(box.cls) == 0:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cx = (x1 + x2) // 2 + pad[0]
                                cy = (y1 + y2) // 2 + pad[1]
                                pt = np.array([[cx, cy]], dtype=np.float32)
                                real = apply_homography(pt, H)
                                if point_in_polygon(real[0], REAL_AREA):
                                    pred_people.append(real[0])
                pred_vector = predict_occupancy(pred_people)
                if fname not in label_data:
                    continue
                gt_dict = label_data[fname]
                gt_vector = [int(gt_dict.get(sid, False)) for sid in seat_ids]
                for p, g, sid in zip(pred_vector, gt_vector, seat_ids):
                    if g == 1 and p == 1:
                        total_stats[sid]["TP"] += 1
                    elif g == 1 and p == 0:
                        total_stats[sid]["FN"] += 1
                    elif g == 0 and p == 1:
                        total_stats[sid]["FP"] += 1
                    else:
                        total_stats[sid]["TN"] += 1

        TP = sum(s["TP"] for s in total_stats.values())
        FP = sum(s["FP"] for s in total_stats.values())
        FN = sum(s["FN"] for s in total_stats.values())
        TN = sum(s["TN"] for s in total_stats.values())
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        conf_results.append((conf_thres, precision, recall, f1))

    # 📈 시각화
    conf_vals = [x[0] for x in conf_results]
    f1_vals = [x[3] for x in conf_results]
    plt.figure(figsize=(8, 5))
    plt.plot(conf_vals, f1_vals, marker='o', label='F1-score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Confidence Threshold')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return conf_results


# ========= 평가 루프 =========
base_dir = "data/scen_output"
label_dir = "data/scen_label/center_camera"

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
    fp_frames = defaultdict(list)
    fn_frames = defaultdict(list)

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
                for view, path in view_paths.items():
                    fn_path = os.path.join(path, fname)
                    if os.path.isfile(fn_path):
                        fn_frames[view].append(fn_path)
            elif g == 0 and p == 1:
                seat_stats[sid]["FP"] += 1
                for view, path in view_paths.items():
                    fp_path = os.path.join(path, fname)
                    if os.path.isfile(fp_path):
                        fp_frames[view].append(fp_path)
            else:
                seat_stats[sid]["TN"] += 1

    visualize_seat_detection_rate(seat_stats)
    show_fp_images(fp_frames, scenario_index)
    show_fn_images(fn_frames, scenario_index)

# evaluate_conf_range(
#     model,
#     base_dir="data/scen_output",
#     label_dir="data/scen_label",
#     seat_ids=seat_ids,
#     apply_homography=apply_homography,
#     point_in_polygon=point_in_polygon,
#     predict_occupancy=predict_occupancy,
#     extract_number=extract_number,
#     homographies_default=homographies_default,
#     homographies_alt=homographies_alt,
#     REAL_AREA=REAL_AREA
# )