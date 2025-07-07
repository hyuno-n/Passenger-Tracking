import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

# 📌 YOLO 모델 로드
model = YOLO("head.pt")

# 📌 데이터 폴더 경로 설정
side_folder = "output_scenario/scen1/camera6_image_raw_flat_single"
back_folder = "output_scenario/scen1/camera8_image_raw_flat_single"

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def load_images_in_order(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")], key=extract_number)
    images = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

side_images = load_images_in_order(side_folder)
back_images = load_images_in_order(back_folder)
num_frames = min(len(side_images), len(back_images))

H1 = np.array([
    [0.312293052, -3.06614997, 923.450556],
    [-0.561703036, -0.589954372, 555.107180],
    [0.000631420942, -0.00462906929, 1.0]
])

H2 = np.array([
    [0.00606036999, 0.215704280, -204.480699],
    [0.431094911, -0.399161955, -156.359721],
    [0.000100314085, -0.00297126407, 1.0]
])

REAL_AREA = np.array([[0, 0], [0, 240], [550, 240], [550, 0]], dtype=np.float32)

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0

def detect_people_in_real_area(image, padding, H):
    results = model(image, verbose=False)
    person_data = []
    img_copy = image.copy()

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                pt_image = np.array([[cx + padding[0], cy + padding[1]]], dtype=np.float32)
                pt_real = apply_homography(pt_image, H)
                if point_in_polygon(pt_real[0], REAL_AREA):
                    person_data.append(pt_real[0])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
    return np.array(person_data, dtype=np.float32), img_copy

def apply_homography(points, H):
    return cv2.perspectiveTransform(np.array([points]), H)[0]

seat_width = 75
seat_height = 50
seat_start_x = 30
seat_start_y = 0

def generate_seat_boxes():
    boxes = []
    for row in range(2):
        num_seats = 7 if row == 0 else 5
        for col in range(num_seats):
            x = seat_start_x + col * seat_width
            y = seat_start_y + row * seat_height
            boxes.append((x, y, seat_width, seat_height))
    for offset in range(2, 5):
        x = seat_start_x + offset * seat_width
        y = 190
        boxes.append((x, y, seat_width, seat_height))
    return boxes

def rotate_for_display(points):
    return np.array([[y, 550 - x] for x, y in points])

def is_person_in_seat(person, seat_box):
    x, y, w, h = seat_box
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    rotated = rotate_for_display(corners)
    x_coords = rotated[:, 0]
    y_coords = rotated[:, 1]
    return (x_coords.min() <= person[0] <= x_coords.max()) and (y_coords.min() <= person[1] <= y_coords.max())

stop_loop = False

def on_key(event):
    global stop_loop
    if event.key == 'q':
        print("👋 종료합니다.")
        stop_loop = True
        plt.close('all')
    else:
        plt.close('all')


for i in range(num_frames):
    if stop_loop:
        break
    print(f"\n🟢 현재 프레임: {i}/{num_frames - 1}")

    img1 = side_images[i]
    img2 = back_images[i]

    padding1 = (300, 150)
    padding2 = (300, 150)

    people1, img1_viz = detect_people_in_real_area(img1, padding1, H1)
    people2, img2_viz = detect_people_in_real_area(img2, padding2, H2)

    if len(people1) < 1 and len(people2) < 1:
        print(f"❌ 검출된 사람이 부족함 (Frame {i})")
        continue

    rotated_people1 = rotate_for_display(people1)
    rotated_people2 = rotate_for_display(people2)

    fig = plt.figure(figsize=(12, 14))

    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=2)
    ax1.imshow(cv2.cvtColor(img1_viz, cv2.COLOR_BGR2RGB), aspect='auto')
    ax1.set_title("Back - YOLO")
    ax1.axis("off")

    ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=3, rowspan=2)
    ax2.imshow(cv2.cvtColor(img2_viz, cv2.COLOR_BGR2RGB), aspect='auto')
    ax2.set_title("Front - YOLO")
    ax2.axis("off")

    ax3 = plt.subplot2grid((4, 4), (0, 3), rowspan=2)
    ax3.set_title("Back Homography Rotated View")
    ax3.scatter(rotated_people1[:, 0], rotated_people1[:, 1], color='red', alpha=0.6)
    ax3.set_xlim(0, 240)
    ax3.set_ylim(0, 550)
    ax3.grid(False)

    ax4 = plt.subplot2grid((4, 4), (2, 3), rowspan=2)
    ax4.set_title("Front Homography Rotated View")
    ax4.scatter(rotated_people2[:, 0], rotated_people2[:, 1], color='blue', alpha=0.6)
    ax4.set_xlim(0, 240)
    ax4.set_ylim(0, 550)
    ax4.invert_xaxis()
    ax4.invert_yaxis()
    ax4.grid(False)

    for ax, people in zip([ax3, ax4], [rotated_people1, rotated_people2]):
        for (x, y, w, h) in generate_seat_boxes():
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            rotated = rotate_for_display(corners)
            occupied = any(is_person_in_seat(p, (x, y, w, h)) for p in people)

            if occupied:
                patch = plt.Polygon(rotated,
                                    fill=True,
                                    edgecolor='gray',
                                    facecolor='red',
                                    alpha=0.3,
                                    linestyle='--',
                                    linewidth=1.5)
            else:
                patch = plt.Polygon(rotated,
                                    fill=False,
                                    edgecolor='gray',
                                    linestyle='--',
                                    linewidth=1.5)
            ax.add_patch(patch)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()