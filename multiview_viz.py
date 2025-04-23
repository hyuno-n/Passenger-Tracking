import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
from matplotlib.widgets import Button

# ===== 설정 =====
BASE_DIR = "data/scen_output"
VIEWS = ["view_-40", "view_0", "view_40"]
VIEW_COLORS = {
    "view_-40": 'blue',
    "view_0": 'green',
    "view_40": 'red'
}
SCENARIOS_TO_VIEW = ["scen2"]  # 원하는 시나리오만 보기
SAVE_OUTPUT = True
SAVE_DIR = "viz_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== YOLO 모델 로드 =====
model = YOLO("runs/detect/add_tire_finetune_freeze/weights/best.pt")

# ===== Homography 행렬 및 패딩 =====
H_map = {
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
REAL_AREA = np.array([[0, 0], [0, 240], [550, 240], [550, 0]], dtype=np.float32)

# ===== 좌석 정의 =====
def generate_seat_boxes():
    boxes = []
    seat_width = 75
    seat_height = 50
    seat_start_x = 30
    seat_start_y = 0

    for row in range(2):
        num_seats = 7 if row == 0 else 5
        for col in range(num_seats):
            x = seat_start_x + col * seat_width
            y = seat_start_y + row * seat_height
            if row == 0 and col in [5, 6]:  # 좌석 6, 7 (index 기준 5, 6)
                y += 20  # y축 위로 20 띄우기
            boxes.append((x, y, seat_width, seat_height))

    rear_offsets = [2 * seat_width, 3 * seat_width + 10, 4 * seat_width + 20]  # 14번과 15번 사이에 추가 여백 10
    for offset in rear_offsets:
        x = seat_start_x + offset
        y = 190
        boxes.append((x, y, seat_width, seat_height))

    return boxes

# ===== YOLO + Homography 적용 =====
def detect_and_project(image, H, padding):
    results = model.predict(source=image, conf=0.5, verbose=False)[0]
    people = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx, cy = (x1 + x2) / 2 + padding[0], (y1 + y2) / 2 + padding[1]
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, H)[0][0]
        if cv2.pointPolygonTest(REAL_AREA, tuple(projected), False) >= 0:
            people.append(projected)
    return np.array(people)

# ===== 시각화 루프용 데이터 =====
frames_to_show = []
for scen in sorted(os.listdir(BASE_DIR)):
    if not scen.startswith("scen") or scen not in SCENARIOS_TO_VIEW:
        continue
    view0_dir = os.path.join(BASE_DIR, scen, "view_0")
    if not os.path.isdir(view0_dir):
        continue
    frame_files = sorted(f for f in os.listdir(view0_dir) if f.endswith(".jpg"))
    for fname in frame_files:
        frames_to_show.append((scen, fname))

# ===== 시각화 함수 =====
current_idx = 0
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
plt.subplots_adjust(bottom=0.2)

btn_ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
btn_ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
btn_prev = Button(btn_ax_prev, '← Prev')
btn_next = Button(btn_ax_next, 'Next →')

text_box = fig.text(0.5, 0.01, '', ha='center')


def update_frame():
    global current_idx, axs
    axs[0].clear()
    axs[1].clear()
    scen, fname = frames_to_show[current_idx]
    scen_path = os.path.join(BASE_DIR, scen)
    original_path = os.path.join(scen_path, fname)
    original_img = cv2.imread(original_path)

    # 3개 view 검출점 + 색상별로 구분
    all_points_by_view = {}
    for view in VIEWS:
        img_path = os.path.join(scen_path, view, fname)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        H, pad = H_map[view]
        people = detect_and_project(img, H, pad)
        all_points_by_view[view] = people

    # 시각화
    axs[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"Original Fisheye Image\n{scen}/{fname}")
    axs[0].axis("off")

    axs[1].set_title("Seat + Head Detection (World Coordinate)")
    axs[1].set_xlim(0, 550)
    axs[1].set_ylim(0, 240)
    axs[1].invert_yaxis()
    axs[1].invert_xaxis()
    axs[1].grid(True)

    seat_boxes = generate_seat_boxes()
    for i, (x, y, w, h) in enumerate(seat_boxes):
        seat_rect = Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black',
                              facecolor='orange', alpha=0.3)
        # 점이 이 좌석 안에 들어오는지 확인
        occupied = False
        for points in all_points_by_view.values():
            for (px, py) in points:
                if x <= px <= x + w and y <= py <= y + h:
                    occupied = True
                    break
            if occupied:
                break
        if occupied:
            seat_rect.set_facecolor('red')
            seat_rect.set_alpha(0.5)
        axs[1].add_patch(seat_rect)
        axs[1].text(x + w / 2, y + h / 2, f"S{i + 1}", ha="center", va="center", fontsize=8)

    for view, points in all_points_by_view.items():
        if points.any():
            px, py = zip(*points)
            axs[1].scatter(px, py, c=VIEW_COLORS[view], label=view, s=60)

    axs[1].legend()
    text_box.set_text(f"{scen} - {fname} ({current_idx + 1}/{len(frames_to_show)})")

    if SAVE_OUTPUT:
        save_path = os.path.join(SAVE_DIR, f"{scen}_{fname.replace('.jpg', '')}.png")
        fig.savefig(save_path)

    fig.canvas.draw_idle()


def next_frame(event):
    global current_idx
    current_idx = (current_idx + 1) % len(frames_to_show)
    update_frame()


def prev_frame(event):
    global current_idx
    current_idx = (current_idx - 1) % len(frames_to_show)
    update_frame()

btn_next.on_clicked(next_frame)
btn_prev.on_clicked(prev_frame)

update_frame()
plt.show()