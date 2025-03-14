import os
import json
import glob
import xml.etree.ElementTree as ET

# ✅ YOLO 라벨이 저장된 폴더 (측면 시점)
yolo_labels_dir = "/home/krri/Desktop/py-ws/boost_track/data/side/labels/"
image_width, image_height = 640, 640  # YOLO는 정규화된 값이므로 실제 크기로 변환 필요

def parse_yolo_labels():
    gt_data = {}

    for label_file in glob.glob(os.path.join(yolo_labels_dir, "*.txt")):
        # ✅ classes.txt 파일 무시
        if "classes.txt" in label_file:
            continue  

        frame_id = os.path.basename(label_file).split(".")[0]  # 파일명에서 프레임 ID 추출

        with open(label_file, "r") as f:
            lines = f.readlines()

        detections = []
        for line in lines:
            data = line.split()
            if data[0] == "0":  # ✅ '0' = `N`은 무시
                continue

            try:
                track_id, x_center, y_center, width, height = map(float, data)
            except ValueError:
                print(f"🚨 YOLO 파일 `{label_file}`에서 변환 오류 발생: {line}")
                continue  # ✅ 변환 오류 발생 시 해당 라인 무시

            # ✅ YOLO 좌표 변환
            x = int((x_center - width / 2) * image_width)
            y = int((y_center - height / 2) * image_height)
            w = int(width * image_width)
            h = int(height * image_height)

            detections.append([x, y, w, h, int(track_id), 0])  # `cam_id = 0`

        if detections:
            gt_data[frame_id] = detections

    return gt_data

# ✅ XML 라벨이 저장된 폴더 (후면 시점)
xml_labels_dir = "/home/krri/Desktop/py-ws/boost_track/data/back/labels/"

EXCLUDED_CLASSES = {"100", "200", "17", "31"}  # ✅ 제외할 객체 ID

def parse_xml_labels():
    gt_data = {}

    for xml_file in glob.glob(os.path.join(xml_labels_dir, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        frame_id = os.path.splitext(root.find("filename").text)[0]  # 파일명에서 프레임 ID 추출
        detections = []

        for obj in root.findall("object"):
            track_id = obj.find("name").text

            if track_id in EXCLUDED_CLASSES:  # ✅ 제외할 객체는 무시
                continue

            # ✅ 바운딩 박스 좌표 변환
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            width = xmax - xmin
            height = ymax - ymin

            detections.append([xmin, ymin, width, height, int(track_id), 1])  # `cam_id = 1`

        if detections:
            gt_data[frame_id] = detections

    return gt_data


# ✅ YOLO + XML 데이터 병합
yolo_gt = parse_yolo_labels()
xml_gt = parse_xml_labels()

gt_test_data = {}

# ✅ 데이터 병합
for frame_id, detections in yolo_gt.items():
    gt_test_data[frame_id] = detections

for frame_id, detections in xml_gt.items():
    if frame_id in gt_test_data:
        gt_test_data[frame_id].extend(detections)
    else:
        gt_test_data[frame_id] = detections

# ✅ 프레임 ID 순서대로 정렬하여 저장
sorted_gt_test_data = {str(k): gt_test_data[str(k)] for k in sorted(map(int, gt_test_data.keys()))}

# ✅ JSON 파일 저장 경로
output_json_path = "/home/krri/Desktop/py-ws/ReST/datasets/BRT/sequence1/output/gt_test.json"

# ✅ 폴더가 없으면 자동 생성
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

# ✅ JSON 파일 저장
with open(output_json_path, "w") as f:
    json.dump(sorted_gt_test_data, f, indent=4)

print(f"✅ 변환 완료: {output_json_path}")
