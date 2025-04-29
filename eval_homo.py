import os
import json
import re
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ✅ 모델 로딩
model = YOLO("runs/detect/add_tire_finetune_freeze/weights/best.pt")

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

# ✅ 좌석 정의
def define_seats():
    seat_width = 75
    seat_start_x = 30
    seats = []
    for col in range(7):
        x = seat_start_x + col * seat_width
        y = 0 if col < 5 else 20
        seats.append((x, y))
    for col in range(5):
        x = seat_start_x + col * seat_width
        y = 50
        seats.append((x, y))
    rear_xs = [
        seat_start_x + 2 * seat_width,
        seat_start_x + 2 * seat_width + 85,
        seat_start_x + 2 * seat_width + 170
    ]
    for x in rear_xs:
        seats.append((x, 180))
    return seats

SEATS = define_seats()
REAL_AREA = np.array([[0, 0], [0, 240], [550, 240], [550, 0]], dtype=np.float32)
SEAT_IDS = [f"S{i}" for i in range(1, 16)]
SAVE_DIR = "results"

def get_seat_size(index):
    return (75, 60) if index >= 12 else (75, 50)

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

def apply_inverse_homography(points, H):
    H_inv = np.linalg.inv(H)
    return cv2.perspectiveTransform(np.array([points], dtype=np.float32), H_inv)[0]

def get_seat_corners(x, y, w, h):
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float32)

def draw_detections_with_inverse_seats(image, results, H, pad, seats=SEATS):
    img = image.copy()

    # 머리 바운딩 박스
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in [x1, y1, x2, y2]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 좌석 기준선 (역투영 → 패딩 보정)
    for idx, (x, y) in enumerate(seats):
        w, h = get_seat_size(idx)
        seat_box = get_seat_corners(x, y, w, h)
        projected_box = apply_inverse_homography(seat_box, H)
        projected_box -= np.array(pad, dtype=np.float32)  # ✅ 여기서 padding 제거
        projected_box = projected_box.astype(int)
        cv2.polylines(img, [projected_box], isClosed=True, color=(0, 255, 0), thickness=2)
        sx, sy = projected_box[0]
        cv2.putText(img, f"S{idx+1}", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return img

def fig_to_image(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

# ✅ 평가 루프
base_dir = "data/front_rear_data"
label_dir = "data/scen_label/front_rear_camera"

def draw_detections_with_inverse_seats(image, results, H, pad, seats=SEATS):
    img = image.copy()
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(seats):
        w, h = get_seat_size(idx)
        corners = get_seat_corners(x, y, w, h)
        projected_box = apply_inverse_homography(corners, H)
        projected_box -= np.array(pad, dtype=np.float32)
        projected_box = projected_box.astype(int)
        cv2.polylines(img, [projected_box], isClosed=True, color=(0, 255, 0), thickness=2)
        sx, sy = projected_box[0]
        cv2.putText(img, f"S{idx+1}", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def draw_flat_seat_layout(seats=SEATS, all_points_by_view=None, title=None, seat_stats=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 240)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_title(title if title else "Seat + Head Detection (World Coordinate)")
    ax.grid(True)

    # 좌석 시각화 (반투명 색상으로 채우기)
    for idx, (x, y) in enumerate(seats):
        w, h = get_seat_size(idx)
        sid = f"S{idx+1}"
        color = 'gray'
        alpha = 0.2
        if seat_stats:
            if seat_stats[sid]['FN'] > 0:
                color = 'blue'
            if seat_stats[sid]['FP'] > 0:
                color = 'red'
            if seat_stats[sid]['FN'] > 0 and seat_stats[sid]['FP'] > 0:
                color = 'purple'
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + 3, y + 15, sid, fontsize=8, color='black')

    # 검출된 head 점 시각화 (view별 색 구분)
    if all_points_by_view:
        colors = {"view_-40": "red", "view_0": "blue", "view_40": "green",
                  "front": "blue", "rear": "orange"}
        for view, points in all_points_by_view.items():
            pxs, pys = zip(*points) if points else ([], [])
            ax.scatter(pxs, pys, c=colors.get(view, "black"), label=f"{view} ({len(points)})", s=20)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_and_save_combined_figure(front_img, rear_img, seat_layout_img, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Front View")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Rear View")
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(seat_layout_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Seat Layout")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_all_scenarios():
    base_dir = "data/front_rear_data"
    label_dir = "data/scen_label/front_rear_camera"
    os.makedirs(SAVE_DIR, exist_ok=True)
    total_stats = {
        cam: {sid: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for sid in SEAT_IDS}
        for cam in ["front", "rear", "combined"]
    }
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
        fp_save_count = defaultdict(int)
        fn_save_count = defaultdict(int)
        save_limit = 3

        for fname in tqdm(filenames, desc=f"🎞 {scen}"):
            all_points_by_view = {}
            all_preds = {}

            for cam in ["front", "rear"]:
                img_path = os.path.join(scen_path, cam, fname)
                if not os.path.exists(img_path):
                    all_preds[cam] = [0] * len(SEATS)  # 없는 경우는 전부 비어있다고 가정
                    continue

                img = cv2.imread(img_path)
                H, pad = homographies[cam]["H"], homographies[cam]["pad"]
                points = detect_heads(img, H, pad)
                all_points_by_view[cam] = points
                all_preds[cam] = predict_occupancy(points)

            # front + rear 통합
            combined = [max(f, r) for f, r in zip(all_preds["front"], all_preds["rear"])]
            all_preds["combined"] = combined

            # Ground truth
            if fname not in gt_data:
                continue
            gt = [int(gt_data[fname].get(sid, False)) for sid in SEAT_IDS]

            # 카메라별로 누적 stats 기록
            for cam in ["front", "rear", "combined"]:
                pred = all_preds[cam]
                for p, g, sid in zip(pred, gt, SEAT_IDS):
                    if p == 1 and g == 1:
                        total_stats[cam][sid]["TP"] += 1
                    elif p == 1 and g == 0:
                        total_stats[cam][sid]["FP"] += 1
                    elif p == 0 and g == 1:
                        total_stats[cam][sid]["FN"] += 1
                    else:
                        total_stats[cam][sid]["TN"] += 1

    for cam in ["front", "rear", "combined"]:
        print(f"\n📊 [{cam.upper()}] 전체 통합 결과:")
        TP = sum([v["TP"] for v in total_stats[cam].values()])
        FP = sum([v["FP"] for v in total_stats[cam].values()])
        FN = sum([v["FN"] for v in total_stats[cam].values()])
        TN = sum([v["TN"] for v in total_stats[cam].values()])
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"TP: {TP} | FP: {FP} | FN: {FN} | TN: {TN}")
        print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1 Score: {f1:.3f}")

        print(f"\n📈 [{cam.upper()}] 좌석별 성능 지표:")
        for sid, val in total_stats[cam].items():
            tp, fp, fn, tn = val["TP"], val["FP"], val["FN"], val["TN"]
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1s = 2 * p * r / (p + r) if (p + r) else 0
            print(f"{sid}: TP={tp}, FP={fp}, FN={fn}, TN={tn} → Precision={p:.3f}, Recall={r:.3f}, F1={f1s:.3f}")


if __name__ == "__main__":
    evaluate_all_scenarios()
