from collections import deque
import numpy as np
import json
import os

# ID 매칭 히스토리 저장 (최근 N 프레임)
ID_MATCH_HISTORY = {}
HISTORY_WINDOW = 10  # 최근 10 프레임 동안 같은 객체를 추적

def compute_mota(fp, fn, idsw, gt_count):
    """
    MOTA (Multi-Object Tracking Accuracy) 계산 (전체 합산 후 계산)
    """
    return max(0, 1 - (fn + fp + idsw) / gt_count) if gt_count > 0 else 0

def compute_hota(association_score, detection_score):
    """
    HOTA (Higher Order Tracking Accuracy) 계산
    """
    return np.sqrt(association_score * detection_score)
def compute_mot_metrics(tracker_results, gt_boxes, frame_idx, matched_results, cam_name, output_dir="tracking_results"):
    """
    프레임별 IDSW, FP, FN, TP 발생 ID를 기록하는 함수
    
    :param tracker_results: Tracker 결과 [[x1, y1, x2, y2, track_id, conf], ...] (track_id는 int형)
    :param gt_boxes: GT 박스 [[xml_id, x1, y1, x2, y2], ...] (xml_id는 str형)
    :param frame_idx: 현재 프레임 인덱스
    :param matched_results: Tracker ID → GT ID 매핑
    :param cam_name: 카메라 이름 (예: 'front', 'side', 'back')
    :param output_dir: 결과 JSON 파일을 저장할 디렉토리 (기본값: 'tracking_results')
    :return: (FP, FN, ID Switches, GT count, ID 유지된 개체 수, TP count, TP ID 리스트)
    """
    global ID_MATCH_HISTORY

    # 특정 cam_name에 따라 GT ID 필터링 조건 설정
    if cam_name == 'side':
        gt_boxes = [g for g in gt_boxes if g[0] != '0']
    elif cam_name == 'back':
        gt_boxes = [g for g in gt_boxes if g[0] not in ['100', '200', '17', '31']]

    # GT ID가 0이 아닌 경우만 남긴 GT ID 집합 생성
    valid_gt_ids = set(int(g[0]) for g in gt_boxes)

    # GT ID 필터링: matched_results에서 GT ID가 유효한 경우만 사용
    matched_results = {t_id: g_id for t_id, g_id in matched_results.items() if g_id in valid_gt_ids}

    # Tracker에서 GT와 매칭된 ID 목록 생성 (valid_track_ids)
    valid_track_ids = set(matched_results.keys())
    # Tracker results에서 GT와 매칭되지 않은 객체(FP)는 유지하도록 수정
    tracker_results = [t for t in tracker_results if t[4] in valid_track_ids]

    total_gt = len(gt_boxes)
    total_tracks = len(tracker_results)

    # TP: 매칭된 Tracker-GT 쌍 수 (matched_results의 개수)
    matched_count = len(matched_results)
    tp_count = matched_count
    tp_ids = list(matched_results.values())

    # FP (False Positives): 매칭되지 않은 Tracker 탐지
    false_positives = list(set(int(t[4]) for t in tracker_results) - set(matched_results.keys()))
    
    # 모든 GT ID 리스트 (중복 포함)
    gt_ids = [int(g[0]) for g in gt_boxes]
    # 매칭된 GT ID 목록 (중복 제거)
    matched_gt_ids = set(matched_results.values())
    # FN (False Negatives): 매칭되지 않은 GT ID들만 카운트
    false_negatives = [gt_id for gt_id in set(gt_ids) if gt_id not in matched_gt_ids]

    # ID Switch 계산 (최근 N 프레임 히스토리와 비교)
    id_switches = 0
    id_switch_list = []  # ID Switch 발생한 GT ID 목록
    correct_id_matches = 0  # ID가 유지된 개체 수

    for track_id, gt_id in matched_results.items():
        if gt_id not in ID_MATCH_HISTORY:
            ID_MATCH_HISTORY[gt_id] = deque(maxlen=HISTORY_WINDOW)
        
        if len(ID_MATCH_HISTORY[gt_id]) > 0:
            if track_id in ID_MATCH_HISTORY[gt_id]:  # ID 유지됨
                correct_id_matches += 1
            else:  # ID 변경됨 → ID Switch 발생
                id_switches += 1
                id_switch_list.append(gt_id)
        # 히스토리에 현재 트랙 ID 추가 (최근 N 프레임 동안 유지)
        ID_MATCH_HISTORY[gt_id].append(track_id)

    # JSON 데이터 저장 (TP ID 리스트도 추가)
    save_id_tracking_results(cam_name, frame_idx, id_switch_list, false_positives, false_negatives, tp_ids, output_dir)

    # 디버깅 출력
    print(f"\n===== Frame {frame_idx} Debug Info ({cam_name}) =====")
    print(f"Total GT Boxes: {total_gt}")
    print(f"Total Tracker Results: {total_tracks}")
    print(f"Matched Pairs (TP): {matched_count}")
    print(f"False Positives (FP): {false_positives} (Count: {len(false_positives)})")
    print(f"False Negatives (FN): {false_negatives} (Count: {len(false_negatives)})")
    print(f"ID Switches: {id_switch_list} (Count: {id_switches})")
    print(f"True Positives (TP): {tp_ids} (Count: {tp_count})")
    if id_switches > 0:
        print(f"ID Switch occurred for GT IDs: {id_switch_list}")
    print(f"Correctly Maintained IDs: {correct_id_matches}")
    print(f"Recent ID Match History: {dict(ID_MATCH_HISTORY)}")
    print("========================================\n")

    return len(false_positives), len(false_negatives), id_switches, total_gt, correct_id_matches, matched_count, tp_count, tp_ids

def save_id_tracking_results(cam_name, frame_idx, id_switch_list, false_positives, false_negatives, tp_ids, output_dir):
    """
    특정 프레임에서 발생한 ID 스위치, FP, FN, TP 정보를 JSON 파일로 저장하는 함수

    :param cam_name: 카메라 이름 (예: 'side', 'back')
    :param frame_idx: 현재 프레임 번호
    :param id_switch_list: ID 스위치가 발생한 GT ID 리스트
    :param false_positives: FP가 발생한 트랙 ID 리스트
    :param false_negatives: FN이 발생한 GT ID 리스트
    :param tp_ids: TP가 발생한 트랙 ID 리스트
    :param output_dir: JSON 저장 경로
    """
    file_path = f"{output_dir}/{cam_name}_tracking_results.json"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # numpy.int64 -> Python int 변환
    frame_idx = int(frame_idx)
    id_switch_list = [int(i) for i in id_switch_list]
    false_positives = [int(i) for i in false_positives]
    false_negatives = [int(i) for i in false_negatives]
    tp_ids = [int(i) for i in tp_ids]

    data[str(frame_idx)] = {
        "ID Switches": id_switch_list,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "True Positives": tp_ids
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ {file_path}에 ID 추적 결과 저장 완료!")

def get_final_mot_metrics(all_metrics):
    """
    모든 프레임의 데이터를 기반으로 최종 MOTA, HOTA, IDF1을 계산

    :param all_metrics: 각 프레임의 (FP, FN, IDSW, GT, ID 유지된 개체 수, 총 매칭 개체 수, TP count) 리스트
    :return: 최종 MOTA, HOTA, IDF1, 총 FP, 총 FN, 총 IDSW, 총 GT, 총 TP
    """
    total_fp = sum(m[0] for m in all_metrics)
    total_fn = sum(m[1] for m in all_metrics)
    total_idsw = sum(m[2] for m in all_metrics)
    total_gt = sum(m[3] for m in all_metrics)
    total_correct_id_matches = sum(m[4] for m in all_metrics)  # ID가 유지된 개체 수
    total_valid_matches = sum(m[5] for m in all_metrics)  # 총 매칭 개체 수 (TP에 해당)
    total_tp = sum(m[6] for m in all_metrics)  # TP 총합

    # 최종 MOTA 계산
    mota = compute_mota(total_fp, total_fn, total_idsw, total_gt)

    # ID Precision (IDP) 및 ID Recall (IDR) 계산 (ID가 변경되지 않은 개체만 고려)
    idp = total_correct_id_matches / (total_valid_matches + 1e-10) if total_valid_matches > 0 else 0
    idr = total_correct_id_matches / (total_gt + 1e-10) if total_gt > 0 else 0

    # 최종 IDF1 계산
    idf1 = 2 * (idp * idr) / (idp + idr + 1e-10) if (idp + idr) > 0 else 0

    # HOTA 계산 (ID Precision과 Recall을 활용)
    hota = compute_hota(idp, idr)

    return mota, hota, idf1, total_fp, total_fn, total_idsw, total_gt, total_tp
