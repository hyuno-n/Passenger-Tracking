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
SCENARIOS_TO_VIEW = ["scen7"]  # 원하는 시나리오만 보기
SAVE_OUTPUT = True
SAVE_DIR = "viz_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== YOLO 모델 로드 =====
model = YOLO("runs/detect/head_finetuned2/weights/best.pt")

# ===== Homography 행렬 및 패딩 =====
H_map = {
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
    results = model.predict(source=image, conf=0.3, verbose=False)[0]
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