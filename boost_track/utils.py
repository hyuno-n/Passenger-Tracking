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

def match_tracker_to_gt(tracker_results, xml_file, iou_threshold=0.5, distance_threshold=100):
    """
    트래커 결과와 GT 데이터를 매칭 (IoU + 거리 기반 보완)

    :param tracker_results: 트래커 결과 [[x1, y1, x2, y2, track_id, conf], ...]
    :param xml_file: GT 데이터가 포함된 XML 파일 경로
    :param iou_threshold: IoU 매칭 기준
    :param distance_threshold: 거리 기반 보완 기준
    :return: 트래커 ID → GT ID 매핑 딕셔너리
    """
    gt_boxes = get_bboxes_from_xml(xml_file)  # GT 박스 로드

    # GT 데이터를 {GT_ID: [x1, y1, x2, y2]} 형태로 변환
    gt_dict = {int(gt[0]): np.array(gt[1:]) for gt in gt_boxes}
    
    track_to_gt_mapping = {}  # 최종 매칭 결과
    unmatched_tracks = list(tracker_results[:, 4].astype(int))  # 매칭되지 않은 트랙 ID 리스트
    unmatched_gt = list(gt_dict.keys())  # 매칭되지 않은 GT ID 리스트

    # 트랙 박스 {track_id → bbox} 변환
    track_dict = {int(track[4]): np.array(track[:4]) for track in tracker_results}

    for track_id, track_box in track_dict.items():
        max_iou = 0  # IoU 기본값 0
        best_gt_id = None

        # IoU 계산
        for gt_id in unmatched_gt:
            iou_value = calculate_iou(track_box, gt_dict[gt_id])

            if iou_value > max_iou:
                max_iou = iou_value
                best_gt_id = gt_id

        # IoU가 임계값 이상이면 매칭
        if max_iou >= iou_threshold:
            track_to_gt_mapping[track_id] = best_gt_id
            unmatched_gt.remove(best_gt_id)  # 매칭된 GT 제거
            unmatched_tracks.remove(track_id)  # 매칭된 트랙 제거

    # IoU로 매칭되지 않은 트랙 → 거리 기반으로 보완
    for track_id in unmatched_tracks[:]:  # 리스트 복사 후 수정
        track_box = track_dict[track_id]
        min_distance = float('inf')
        best_gt_id = None

        for gt_id in unmatched_gt:
            distance = calculate_euclidean_distance(track_box, gt_dict[gt_id])

            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                best_gt_id = gt_id

        if best_gt_id is not None:
            track_to_gt_mapping[track_id] = best_gt_id
            unmatched_gt.remove(best_gt_id)  # 매칭된 GT 제거
            unmatched_tracks.remove(track_id)  # 매칭된 트랙 제거
    return track_to_gt_mapping
