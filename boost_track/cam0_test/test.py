import os
import cv2
import glob
import random
import xml.etree.ElementTree as ET


# 경로 설정
image_dir = "data/back/imgs/"
label_txt_dir = "cam0/labels/"  # YOLO 형식 라벨
label_xml_dir = "data/back/labels/"  # XML 형식 라벨
output_video = "output.avi"

# 이미지와 라벨 파일 읽기
def sort_key(file_path):
    """숫자를 기준으로 파일 이름 정렬"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return int(base_name)  # 파일 이름이 숫자로 되어 있다고 가정


image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")), key=sort_key)
label_txt_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(label_txt_dir, "*.txt"))}
label_xml_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(label_xml_dir, "*.xml"))}

# 출력 동영상 설정
frame_width, frame_height = 1920, 1080  # 해상도 설정 (이미지 해상도에 맞게 수정)
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 색상 설정 (랜덤 색상)
object_colors = {}
def get_color(object_id):
    """객체 ID에 대한 랜덤 색상 반환"""
    if object_id not in object_colors:
        object_colors[object_id] = [random.randint(0, 255) for _ in range(3)]
    return object_colors[object_id]


# YOLO 형식 바운딩 박스 그리기
def draw_boxes_from_txt(image, label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    height, width, _ = image.shape
    for line in lines:
        parts = line.strip().split()
        object_id = int(parts[0])  # 객체 ID
        x_center, y_center, w, h = map(float, parts[1:])

        # YOLO 좌표 -> 픽셀 좌표 변환
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # 박스와 객체 ID 표시
        color = get_color(object_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"ID: {object_id} (YOLO)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# XML 형식 바운딩 박스 그리기
def draw_boxes_from_xml(image, label_path):
    tree = ET.parse(label_path)
    root = tree.getroot()
    
    height, width, _ = image.shape

    for obj in root.findall('object'):
        object_id = obj.find('name').text  # 객체 ID (XML에서는 보통 클래스 이름이 들어감)
        bbox = obj.find('bndbox')
        
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)

        # XML은 XML 고유 색상 사용 (다른 색상 구별)
        color = get_color(object_id + "_xml")  # XML은 다른 색상 유지

        # 박스와 객체 ID 표시
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"ID: {object_id} (XML)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# 모든 이미지 처리 및 영상 저장
for image_file in image_files:
    base_name = os.path.splitext(os.path.basename(image_file))[0]

    # 이미지 읽기
    img = cv2.imread(image_file)
    
    # # YOLO 라벨 처리
    # if base_name in label_txt_files:
    #     label_file = label_txt_files[base_name]
    #     draw_boxes_from_txt(img, label_file)
    
    # XML 라벨 처리
    if base_name in label_xml_files:
        label_file = label_xml_files[base_name]
        draw_boxes_from_xml(img, label_file)

    # 프레임 크기 조정 (필요시 생략 가능)
    img_resized = cv2.resize(img, (frame_width, frame_height))

    # 동영상에 추가
    video_writer.write(img_resized)

# 자원 해제
video_writer.release()
print(f"✅ 동영상 저장 완료: {output_video}")
