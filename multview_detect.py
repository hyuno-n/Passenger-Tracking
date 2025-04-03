from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# YOLO 모델 로드
model = YOLO("head.pt")

# 각도 폴더 정의
angles = [-40, 0, 40]
base_dir = Path("camera2_image_raw_flat_multi")
folders = {angle: base_dir / f"view_{angle}" for angle in angles}

# 이미지 확장자
valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]

# 기준은 view_0 폴더 → 파일 이름 리스트
reference_folder = folders[0]
img_names = sorted([f.name for f in reference_folder.iterdir() if f.suffix.lower() in valid_exts])

# 프레임별 순회
for filename in img_names:
    print(f"\n[INFO] Processing frame: {filename}")
    
    for angle in angles:
        img_path = folders[angle] / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Cannot read: {img_path}")
            continue

        # YOLO 탐지
        results = model.predict(source=img, conf=0.3, save=False, verbose=False)[0]

        # 박스 그리기
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 이미지 보여주기 (하나씩)
        window_name = f"{filename} - Angle {angle}"
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        if key == 27:  # ESC 눌렀을 때 전체 종료
            print("[INFO] ESC pressed. Exiting.")
            cv2.destroyAllWindows()
            exit()
