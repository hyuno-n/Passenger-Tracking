import numpy as np
from utils import calculate_iou

def compute_mota(fp, fn, idsw, gt_count):
    """
    MOTA (Multi-Object Tracking Accuracy) 계산
    """
    return 1 - (fp + fn + idsw) / gt_count if gt_count > 0 else 0

def compute_idf1(gt_matches, tracker_matches):
    """
    IDF1 (Identity F1 Score) 계산
    """
    precision = gt_matches / (tracker_matches + 1e-10)
    recall = gt_matches / (gt_matches + 1e-10)
    return 2 * (precision * recall) / (precision + recall + 1e-10)

def compute_hota(association_score, detection_score):
    """
    HOTA (Higher Order Tracking Accuracy) 계산
    """
    return np.sqrt(association_score * detection_score)

def compute_mot_metrics(tracker_results, gt_boxes, frame_idx, matched_results):
    """
    주어진 프레임의 YOLO 탐지 결과와 GT 매칭을 기반으로 MOTA, MOTP, IDF1 지표를 계산
    
    :param yolo_boxes: YOLO 탐지 결과 [[x1, y1, x2, y2, conf], ...]
    :param gt_boxes: GT 박스 [[xml_id, x1, y1, x2, y2], ...]
    :param frame_idx: 현재 프레임 인덱스
    :param matched_results: YOLO 탐지 ID → GT ID 매핑 (match_yolo_to_gt 결과)
    :return: MOTA, MOTP, IDF1 계산을 위한 값 (FP, FN, ID_SWITCH, IoU 합산값)
    """
    total_gt = len(gt_boxes)
    total_tracks = len(tracker_results)

    # Track ID 기반 박스 저장
    track_dict = {int(track[4]): track[:4] for track in tracker_results}
    
    # 매칭된 YOLO-GT 개수
    matched_count = len(matched_results)
    
    # FP (False Positives): 매칭되지 않은 YOLO 탐지
    false_positives = total_tracks - matched_count
    
    # FN (False Negatives): 매칭되지 않은 GT
    false_negatives = total_gt - matched_count
    
    # ID Switch 계산 (이전 프레임과 비교)
    id_switches = 0
    if frame_idx > 0 and hasattr(compute_mot_metrics, "prev_matched_results"):
        prev_results = compute_mot_metrics.prev_matched_results  # 이전 프레임 {Track ID → GT ID}

        for track_id, gt_id in matched_results.items():
            if track_id in prev_results and prev_results[track_id] != gt_id:
                id_switches += 1

    # IoU 기반 MOTP (Multi-Object Tracking Precision) 계산
    total_iou = 0
    valid_matches = 0
    
    for track_id, gt_id in matched_results.items():
        if track_id in track_dict and gt_id in gt_boxes:
            tracker_box = track_dict[track_id]
            gt_box = gt_boxes[gt_id]

            iou = calculate_iou(tracker_box, gt_box)
            total_iou += iou
            valid_matches += 1

    motp = total_iou / valid_matches if valid_matches > 0 else 0

    # MOTA (Multiple Object Tracking Accuracy) 계산 최소 0으로 제한
    mota = max(0, 1 - (false_negatives + false_positives + id_switches) / total_gt)

    # IDF1 (Identification F1 Score) 계산
    precision = matched_count / (matched_count + false_positives) if (matched_count + false_positives) > 0 else 0
    recall = matched_count / (matched_count + false_negatives) if (matched_count + false_negatives) > 0 else 0
    idf1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 이전 매칭 결과 저장 (ID Switch 계산용)
    compute_mot_metrics.prev_matched_results = matched_results

    return mota, motp, idf1, false_positives, false_negatives, id_switches

