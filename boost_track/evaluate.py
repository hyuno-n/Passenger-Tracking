from collections import deque
import numpy as np

# ID 매칭 히스토리 저장 (최근 N 프레임)
ID_MATCH_HISTORY = {}
HISTORY_WINDOW = 10  # 최근 10 프레임 동안 같은 객체를 추적

def compute_mota(fp, fn, idsw, gt_count):
    """
    MOTA (Multi-Object Tracking Accuracy) 계산 (전체 합산 후 계산)
    """
    return 1 - (fp + fn + idsw) / gt_count if gt_count > 0 else 0

def compute_hota(association_score, detection_score):
    """
    HOTA (Higher Order Tracking Accuracy) 계산
    """
    return np.sqrt(association_score * detection_score)

def compute_mot_metrics(tracker_results, gt_boxes, frame_idx, matched_results):
    """
    주어진 프레임의 Tracker 결과와 GT 매칭을 기반으로 MOTA, HOTA, IDF1 지표를 계산 (전체 합산 방식)
    
    :param tracker_results: Tracker 결과 [[x1, y1, x2, y2, track_id, conf], ...]
    :param gt_boxes: GT 박스 [[xml_id, x1, y1, x2, y2], ...]
    :param frame_idx: 현재 프레임 인덱스
    :param matched_results: Tracker ID → GT ID 매핑
    :return: FP, FN, ID Switches, GT count, ID 유지된 개체 수, 총 매칭 개수
    """
    global ID_MATCH_HISTORY

    total_gt = len(gt_boxes)
    total_tracks = len(tracker_results)

    # 매칭된 Tracker-GT 개수
    matched_count = len(matched_results)
    
    # FP (False Positives): 매칭되지 않은 Tracker 탐지
    false_positives = total_tracks - matched_count
    
    # FN (False Negatives): 매칭되지 않은 GT
    false_negatives = len(gt_boxes) - matched_count
    
    # ID Switch 계산 (최근 N 프레임 히스토리와 비교)
    id_switches = 0
    id_switch_list = []  # ID Switch 발생한 Track ID 목록
    correct_id_matches = 0  # ID가 유지된 개체 수

    for track_id, gt_id in matched_results.items():
        if gt_id not in ID_MATCH_HISTORY:
            ID_MATCH_HISTORY[gt_id] = deque(maxlen=HISTORY_WINDOW)
        
        if len(ID_MATCH_HISTORY[gt_id]) > 0:
            if track_id in ID_MATCH_HISTORY[gt_id]:  # ID 유지됨
                correct_id_matches += 1
            else:  # ID가 변경됨 → ID Switch 발생
                id_switches += 1
                id_switch_list.append(track_id)

        # 히스토리에 현재 트랙 ID 추가 (최근 `N` 프레임 동안 유지)
        ID_MATCH_HISTORY[gt_id].append(track_id)

    # 디버깅 출력
    print(f"\n===== Frame {frame_idx} Debug Info =====")
    print(f"Total GT Boxes: {total_gt}")
    print(f"Total Tracker Results: {total_tracks}")
    print(f"Matched Pairs: {matched_count}")
    print(f"False Positives (FP): {false_positives}")
    print(f"False Negatives (FN): {false_negatives}")
    print(f"ID Switches: {id_switches}")
    if id_switches > 0:
        print(f"ID Switch occurred for Track IDs: {id_switch_list}")
    print(f"Correctly Maintained IDs: {correct_id_matches}")
    print(f"Recent ID Match History: {dict(ID_MATCH_HISTORY)}")
    print("========================================\n")

    return false_positives, false_negatives, id_switches, total_gt, correct_id_matches, matched_count

def get_final_mot_metrics(all_metrics):
    """
    모든 프레임의 데이터를 기반으로 최종 MOTA, HOTA, IDF1을 계산

    :param all_metrics: 각 프레임의 (FP, FN, IDSW, GT, ID 유지된 개체 수, 총 매칭 개수) 리스트
    :return: 최종 MOTA, HOTA, IDF1, 총 FP, 총 FN, 총 IDSW
    """
    total_fp = sum(m[0] for m in all_metrics)
    total_fn = sum(m[1] for m in all_metrics)
    total_idsw = sum(m[2] for m in all_metrics)
    total_gt = sum(m[3] for m in all_metrics)
    total_correct_id_matches = sum(m[4] for m in all_metrics)  # ID가 유지된 개체 수
    total_valid_matches = sum(m[5] for m in all_metrics)  # 총 매칭 개수

    # 최종 MOTA 계산
    mota = max(0, 1 - (total_fn + total_fp + total_idsw) / total_gt) if total_gt > 0 else 0

    # ID Precision (IDP) 및 ID Recall (IDR) 계산 (ID가 변경되지 않은 개체만 고려)
    idp = total_correct_id_matches / (total_valid_matches + 1e-10) if total_valid_matches > 0 else 0
    idr = total_correct_id_matches / (total_gt + 1e-10) if total_gt > 0 else 0

    # 최종 IDF1 계산
    idf1 = 2 * (idp * idr) / (idp + idr + 1e-10) if (idp + idr) > 0 else 0

    # HOTA 계산 (ID Precision과 Recall을 활용)
    hota = compute_hota(idp, idr)

    return mota, hota, idf1, total_fp, total_fn, total_idsw
