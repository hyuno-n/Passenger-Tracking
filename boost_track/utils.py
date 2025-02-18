import glob
import os

import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List


def write_results_no_score(filename, results):
    """Writes results in MOT style to filename."""
    save_format = "{frame},{id},{x1},{y1},{w},{h},{c},-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, conf in results:
            for tlwh, track_id, c in zip(tlwhs, track_ids, conf):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    c=round(c, 2)
                )
                f.write(line)


def filter_targets(online_targets, aspect_ratio_thresh, min_box_area):
    """Removes targets not meeting threshold criteria.

    Returns (list of tlwh, list of ids).
    """
    online_tlwhs = []
    online_ids = []
    online_conf = []
    for t in online_targets:
        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
        tid = t[4]
        tc = t[5]
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_conf.append(tc)
    return online_tlwhs, online_ids, online_conf


def dti(txt_path, save_path, n_min=25, n_dti=20):
    def dti_write_results(filename, results):
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(filename, "w") as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)

    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    # breakpoint()
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]  ## To better play along with windows paths
        print(seq_name)
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        tracklets_to_remove = []
        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            # for idx in range(len(tracklet)):
            #     print(tracklet[idx])
                # print(tracklet[:, 0])
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / (
                                right_frame - left_frame
                            ) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)


def calculate_iou(boxA, boxB):
    """
    IoU(Intersection over Union) 계산
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / (boxA_area + boxB_area - inter_area + 1e-10)

def calculate_centroid(box):
    """바운딩 박스 중심 좌표 계산"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_euclidean_distance(box1, box2):
    """유클리드 거리 계산"""
    c1 = calculate_centroid(box1)
    c2 = calculate_centroid(box2)
    return np.linalg.norm(np.array(c1) - np.array(c2))

def get_bboxes_from_xml(xml_file):
    """XML 파일에서 bounding box 정보 추출"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    
    for obj in root.findall('.//object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bboxes.append((name, xmin, ymin, xmax, ymax))
    
    return sorted(bboxes) 


def map_yolo_to_xml_labels(yolo_boxes, xml_file):
    """
    YOLO 탐지 결과와 XML 파일의 GT 박스를 비교하여 IOU가 가장 높은 것을 매핑
    
    Args:
        yolo_boxes: YOLO 탐지 결과 [x1, y1, x2, y2]
        xml_file: XML 파일 경로
    
    Returns:
        Dict[int, str]: YOLO 탐지 인덱스 → XML 객체 이름 매핑
    """
    xml_boxes = get_bboxes_from_xml(xml_file)
    yolo_label_mapping = {}
    
    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        max_iou = 0
        best_xml_name = None
        
        for xml_box in xml_boxes:
            x1, y1, x2, y2 = xml_box[1:]
            xml_name = xml_box[0]
            
            iou_value = calculate_iou(yolo_box, [x1, y1, x2, y2])
            
            if iou_value > max_iou:
                max_iou = iou_value
                best_xml_name = xml_name
        
        if max_iou >= 0.5 and best_xml_name is not None:
            yolo_label_mapping[yolo_idx] = best_xml_name
    print("yolomapping",yolo_label_mapping)
    return yolo_label_mapping

def match_yolo_to_gt(yolo_boxes, xml_file, iou_threshold=0.5, distance_threshold=100):
    """
    YOLO 탐지 결과와 GT 데이터를 매칭 (IoU + 거리 기반 보완)
    
    :param yolo_boxes: YOLO 탐지 결과 [[x1, y1, x2, y2, conf], ...]
    :param gt_boxes: GT 박스 [[xml_id, x1, y1, x2, y2], ...]
    :param iou_threshold: IoU 매칭 기준
    :param distance_threshold: 거리 기반 보완 기준
    :return: YOLO 탐지 인덱스 → GT ID 매핑 딕셔너리
    """
    gt_boxes = get_bboxes_from_xml(xml_file)
    yolo_to_gt_mapping = {}
    unmatched_yolo = []
    unmatched_gt = list(range(len(gt_boxes)))

    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        max_iou = 0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            gt_bbox = gt_box[1:]  # GT 좌표
            iou_value = calculate_iou(yolo_box[:4], gt_bbox)

            if iou_value > max_iou:
                max_iou = iou_value
                best_gt_idx = gt_idx

        # IoU가 임계값을 넘으면 매칭
        if max_iou >= iou_threshold:
            yolo_to_gt_mapping[yolo_idx] = gt_boxes[best_gt_idx][0]
            unmatched_gt.remove(best_gt_idx)  # 매칭된 GT 제거
        else:
            unmatched_yolo.append(yolo_idx)  # 매칭 실패한 YOLO 박스 기록

    # IoU로 매칭되지 않은 YOLO 탐지 → 거리 기반으로 보완
    for yolo_idx in unmatched_yolo:
        yolo_box = yolo_boxes[yolo_idx][:4]
        min_distance = float('inf')
        best_gt_idx = None

        for gt_idx in unmatched_gt:
            gt_box = gt_boxes[gt_idx][1:]
            distance = calculate_euclidean_distance(yolo_box, gt_box)

            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            yolo_to_gt_mapping[yolo_idx] = gt_boxes[best_gt_idx][0]
            unmatched_gt.remove(best_gt_idx)

    return yolo_to_gt_mapping


def match_detections_with_xml(box_mapping: Dict, xml_result: List , track_boxes : List) -> Dict:
    """ 
    YOLO 탐지 순서와 XML ID 간의 매핑을 분석
    
    Args:
    - box_mapping: YOLO 탐지 순서와 : [track_id , x1,y1,x2,y2,conf]
    - xml_result: XML에서 추출한 박스 정보 [xml_id, x1,y1,x2,y2]
    
    Returns:
    - 매핑 결과 딕셔너리
      {
        'yolo_to_tracking': {yolo_idx: tracking_id},
        'yolo_to_xml': {yolo_idx: xml_id},
        'tracking_to_xml': {tracking_id: xml_id}
      }
    """
    # XML ID 추출 (name에서 숫자 추출)
    xml_id_mapping = {}
    for idx, (name, xmin, ymin, xmax, ymax) in enumerate(xml_result):
        try:
            xml_id = int(''.join(filter(str.isdigit, name)))
            xml_id_mapping[idx] = xml_id
        except:
            # ID를 추출할 수 없는 경우 스킵
            pass
    
    # IoU를 사용하여 YOLO 박스와 XML 박스 매핑
    yolo_to_xml = {}
    for yolo_idx, (track_info) in box_mapping.items():
        tracking_id, x1, y1, x2, y2, conf = track_info
        yolo_box = [x1, y1, x2, y2]
        
        best_iou = 0
        best_xml_idx = None
        for xml_idx, (name, xmin, ymin, xmax, ymax) in enumerate(xml_result):
            xml_box = [xmin, ymin, xmax, ymax]
            iou = calculate_iou(yolo_box, xml_box)
            
            if iou > best_iou:
                best_iou = iou
                best_xml_idx = xml_idx

        if best_iou >= 0.7: # 해당코드가 문제가 발생할수도있음 만약에 XML박스와 yolo박스가 임계치가 만족하지않는다면
            yolo_to_xml[yolo_idx] = best_xml_idx
    
    # 최종 매핑 결과 생성
    """
    YOLO_IDX 탐지순서 : 부여받은 Tracking_ID (가변)
    YOLO_IDX : XML_IDX (불변)
    Tracking_ID : XML_ID  
    
    """
    mapping_result = {
        'yolo_to_tracking': {k: v[0] for k, v in box_mapping.items()}, # yolo_IDX  : tracking_ID
        'yolo_to_xml': yolo_to_xml, # yolo_IDX : XML_IDX
        'tracking_to_xml': {} # tracking ID : XML_ID 
    }
    
    # tracking_to_xml 매핑 추가
    for yolo_idx, tracking_id in mapping_result['yolo_to_tracking'].items():
        if yolo_idx in mapping_result['yolo_to_xml']:
            xml_idx = mapping_result['yolo_to_xml'][yolo_idx]
            if xml_idx in xml_id_mapping:
                mapping_result['tracking_to_xml'][tracking_id] = xml_id_mapping[xml_idx]
    
    return mapping_result