import json

def load_json(file_path):
    """ JSON 파일을 로드하는 함수 """
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_overlap_metrics(side_results, back_results):
    """ 
    두 JSON 데이터에서 같은 프레임 ID에 대해 
    - ID Switches, False Positives, False Negatives의 중복도를 계산하고,
    - 한 시점에서 False Negatives인 객체가 다른 시점에서는 True Positives로 검출된 경우를 계산함.
    
    각 프레임의 JSON 데이터에는 다음 키들이 있어야 합니다:
      - "ID Switches": 해당 프레임에서 발생한 ID 스위치 GT id 리스트
      - "False Positives": FP 트랙 id 리스트
      - "False Negatives": FN (미검출) GT id 리스트
      - "True Positives": TP로 검출된 GT id 리스트
      
    반환:
      overlap_counts, total_counts, overlap_percentages, fn_tp_conversion_count, conversion_details
    """
    overlap_counts = {"ID Switches": 0, "False Positives": 0, "False Negatives": 0}
    total_counts = {"ID Switches": 0, "False Positives": 0, "False Negatives": 0}
    overlapping_details = {}  # 프레임별 중복 정보 저장
    fn_tp_conversion_count = 0  # FN->TP 전환된 경우 총합
    conversion_details = {}      # 프레임별 FN->TP 전환 상세 정보
    
    common_frames = set(side_results.keys()).intersection(set(back_results.keys()))
    
    for frame in common_frames:
        side_data = side_results[frame]
        back_data = back_results[frame]
        
        frame_overlap = {}
        # 기존: ID Switches, FP, FN에 대해 중복 계산
        for key in ["ID Switches", "False Positives", "False Negatives"]:
            side_ids = set(side_data.get(key, []))
            back_ids = set(back_data.get(key, []))
            
            total_counts[key] += len(side_ids) + len(back_ids)
            common_ids = side_ids.intersection(back_ids)
            overlap_counts[key] += len(common_ids)
            
            if common_ids:
                frame_overlap[key] = list(common_ids)
        
        # FN->TP 전환 계산:
        # 각 프레임에서 "False Negatives"와 "True Positives" 키를 활용하여,
        # 예를 들어, side view에서 FN인 GT id가 back view의 TP 리스트에 있다면 이를 전환된 것으로 간주.
        side_fn = set(side_data.get("False Negatives", []))
        back_fn = set(back_data.get("False Negatives", []))
        side_tp = set(side_data.get("True Positives", []))
        back_tp = set(back_data.get("True Positives", []))
        
        # side에서 FN인데, back에서는 TP로 검출된 경우
        side_fn_to_tp = side_fn.intersection(back_tp)
        # back에서 FN인데, side에서는 TP로 검출된 경우
        back_fn_to_tp = back_fn.intersection(side_tp)
        
        frame_conversion = len(side_fn_to_tp) + len(back_fn_to_tp)
        fn_tp_conversion_count += frame_conversion
        
        if frame_conversion > 0:
            frame_overlap["FN->TP"] = list(side_fn_to_tp.union(back_fn_to_tp))
            conversion_details[frame] = {"FN->TP": list(side_fn_to_tp.union(back_fn_to_tp))}
        
        if frame_overlap:
            overlapping_details[frame] = frame_overlap

    overlap_percentages = {
        key: (overlap_counts[key] / total_counts[key] * 100 if total_counts[key] > 0 else 0)
        for key in overlap_counts
    }
    
    return overlap_counts, total_counts, overlap_percentages, fn_tp_conversion_count, conversion_details

# 파일 경로 설정
side_file_path = "./tracking_results/side_tracking_results.json"
back_file_path = "./tracking_results/back_tracking_results.json"

# JSON 데이터 로드
side_data = load_json(side_file_path)
back_data = load_json(back_file_path)

# 중복도 및 FN->TP 전환 케이스 계산
overlap_counts, total_counts, overlap_percentages, fn_tp_conversion_count, conversion_details = compute_overlap_metrics(side_data, back_data)

# 결과 출력
print("=== 중복된 IDSW, FP, FN 개수 (같은 프레임에서만) ===")
print(overlap_counts)

print("\n=== 전체 IDSW, FP, FN 개수 ===")
print(total_counts)

print("\n=== 중복도 (퍼센트) ===")
print(overlap_percentages)

print("\n=== FN에서 TP로 전환된 경우 개수 ===")
print(fn_tp_conversion_count)

# print("\n=== 프레임별 FN->TP 전환 상세 정보 ===")
# for frame, details in conversion_details.items():
#     print(f"🔹 프레임 {frame}:")
#     for key, ids in details.items():
#         print(f"  - {key}: {ids}")
