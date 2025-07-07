import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ✅ 모델 로드
model = YOLO("runs/detect/head_finetuned_freeze2/weights/best.pt")

# ✅ 테스트할 이미지 경로
image_path = "frame_center.jpg"

# ✅ 이미지 로드
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ✅ 예측 수행
results = model.predict(source=img, conf=0.3, save=False, verbose=False)[0]

# ✅ 결과 시각화
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf)
    cls = int(box.cls)

    # 바운딩 박스 그리기
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 중심점 + confidence 표시
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img_rgb, (cx, cy), 4, (0, 255, 0), -1)
    label = f"head {conf:.2f}"
    cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# ✅ 결과 출력
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title("YOLOv8 Head Detection on Fisheye Image")
plt.axis("off")
plt.tight_layout()
plt.show()


def show_fp_images_by_view(fp_image_dir, sample_per_view=3):
    """
    view별로 오탐(FP) 이미지를 sample_per_view장씩 불러와 한 창에 띄우는 함수
    """
    view_map = {"view_-40": [], "view_0": [], "view_40": []}

    for fname in os.listdir(fp_image_dir):
        if fname.endswith(".jpg"):
            for view in view_map:
                if view in fname:
                    view_map[view].append(os.path.join(fp_image_dir, fname))

    # 각 view에서 최대 sample_per_view개씩 선택
    for view in view_map:
        view_map[view] = sorted(view_map[view])[:sample_per_view]

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (view, paths) in enumerate(view_map.items()):
        combined = None
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if combined is None:
                combined = img
            else:
                combined = cv2.vconcat([combined, img])
        if combined is not None:
            axes[idx].imshow(combined)
            axes[idx].set_title(f"{view}", fontsize=14)
            axes[idx].axis("off")
        else:
            axes[idx].set_title(f"{view} (no images)", fontsize=14)
            axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

# 사용 예시:
show_fp_images_by_view("data/scen_output/scen3", sample_per_view=3)
