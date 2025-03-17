import json

def load_json(file_path):
    """JSON 파일을 로드하는 함수"""
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_overlap_metrics(side_results, back_results):
    """ 
    두 JSON 데이터에서 같은 프레임 ID에 대해 
    - "ID Switches", "False Positives", "False Negatives"의 중복도를 계산하고,
    - 한 시점에서 FN인 객체가 다른 시점에서는 True Positives(TP)로 검출된 경우를 계산함.
    
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
    fn_tp_conversion_count = 0  # FN→TP 전환된 경우 총합
    conversion_details = {}      # 프레임별 FN→TP 전환 상세 정보
    
    common_frames = set(side_results.keys()).intersection(set(back_results.keys()))

    side_conversion, back_conversion = 0, 0

    for frame in common_frames:
        side_data = side_results[frame]
        back_data = back_results[frame]
        
        frame_overlap = {}
        # "ID Switches", "False Positives", "False Negatives"의 중복 계산
        for key in ["ID Switches", "False Positives", "False Negatives"]:
            side_ids = set(side_data.get(key, []))
            back_ids = set(back_data.get(key, []))
            
            total_counts[key] += len(side_ids) + len(back_ids)
            common_ids = side_ids.intersection(back_ids)
            overlap_counts[key] += len(common_ids)
            
            if common_ids:
                frame_overlap[key] = list(common_ids)
        
        # FN→TP 전환 계산:
        # 예: side view에서 FN인 GT id가 back view의 "True Positives" 리스트에 있다면 전환된 것으로 간주
        side_fn = set(side_data.get("False Negatives", []))
        back_fn = set(back_data.get("False Negatives", []))
        side_tp = set(side_data.get("True Positives", []))
        back_tp = set(back_data.get("True Positives", []))
        
        # side에서 FN인데, back에서는 TP로 검출된 경우
        side_fn_to_tp = side_fn.intersection(back_tp)
        # back에서 FN인데, side에서는 TP로 검출된 경우
        back_fn_to_tp = back_fn.intersection(side_tp)
        
        frame_conversion = len(side_fn_to_tp) + len(back_fn_to_tp)
        side_conversion += len(side_fn_to_tp)
        back_conversion += len(back_fn_to_tp)
        
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

    return overlap_counts, total_counts, overlap_percentages, fn_tp_conversion_count, conversion_details, side_conversion, back_conversion


def compute_idsw_complement_metrics(side_results, back_results):
    """
    두 JSON 데이터에서 각 프레임에 대해, 한 시점에서 발생한 ID Switch (IDSW) 이벤트에 대해,
    다른 시점에서 해당 GT 객체가 안정적으로 추적되어 True Positives(TP)에 포함된 경우(보완된 경우)를 계산한다.
    
    예: side view에서 IDSW가 발생한 GT id가 back view에서는 IDSW로 기록되지 않고 대신 TP로 검출되었다면,
    이를 보완된(IDSW 보완) 경우로 카운트한다.
    
    반환:
      - total_complement: 모든 프레임의 보완된 IDSW 개수 총합
      - complement_details: 프레임별 보완된 IDSW 상세 정보 
          예: {frame_id: {"Side Complement": [gt_id1, gt_id2, ...],
                           "Back Complement": [gt_id3, ...]}}
    """
    total_complement = 0
    complement_details = {}
    common_frames = set(side_results.keys()).intersection(set(back_results.keys()))
    
    for frame in common_frames:
        side_data = side_results.get(frame, {})
        back_data = back_results.get(frame, {})
        
        side_idsw = set(side_data.get("ID Switches", []))
        back_idsw = set(back_data.get("ID Switches", []))
        side_tp = set(side_data.get("True Positives", []))
        back_tp = set(back_data.get("True Positives", []))
        
        # side view에서 IDSW 발생했으나, back view에서는 해당 GT id가 TP로 검출된 경우
        side_complement = {gt_id for gt_id in side_idsw if (gt_id not in back_idsw) and (gt_id in back_tp)}
        # back view에서 IDSW 발생했으나, side view에서는 해당 GT id가 TP로 검출된 경우
        back_complement = {gt_id for gt_id in back_idsw if (gt_id not in side_idsw) and (gt_id in side_tp)}
        
        frame_complement_count = len(side_complement) + len(back_complement)
        total_complement += frame_complement_count
        
        if frame_complement_count > 0:
            complement_details[frame] = {
                "Side Complement": list(side_complement),
                "Back Complement": list(back_complement)
            }
    
    return total_complement, complement_details


def compute_fn_tp_with_idsw(conversion_details, side_results, back_results):
    """
    FN에서 TP로 전환된 객체 중 IDSW가 발생한 경우를 확인하는 함수.
    
    - conversion_details: FN → TP로 변환된 객체 목록 (프레임별)
    - side_results, back_results: IDSW 정보를 포함한 원본 데이터
    
    반환:
      - fn_tp_with_idsw_count: IDSW가 발생한 FN→TP 전환 케이스 총합
      - fn_tp_with_idsw_details: 프레임별 상세 정보
    """
    fn_tp_with_idsw_count = 0
    fn_tp_with_idsw_details = {}
    
    for frame, data in conversion_details.items():
        fn_tp_ids = set(data.get("FN->TP", []))
        
        # 해당 프레임의 IDSW 목록 가져오기
        side_idsw = set(side_results.get(frame, {}).get("ID Switches", []))
        back_idsw = set(back_results.get(frame, {}).get("ID Switches", []))
        
        # FN→TP 객체 중 IDSW가 발생한 경우 확인
        fn_tp_with_idsw = fn_tp_ids.intersection(side_idsw.union(back_idsw))
        
        if fn_tp_with_idsw:
            fn_tp_with_idsw_count += len(fn_tp_with_idsw)
            fn_tp_with_idsw_details[frame] = list(fn_tp_with_idsw)
    
    return fn_tp_with_idsw_count, fn_tp_with_idsw_details

# 파일 경로 설정
side_file_path = "./tracking_results/side_tracking_results.json"
back_file_path = "./tracking_results/back_tracking_results.json"

# JSON 데이터 로드
side_data = load_json(side_file_path)
back_data = load_json(back_file_path)

# 중복도 및 FN→TP 전환 케이스 계산
overlap_counts, total_counts, overlap_percentages, fn_tp_conversion_count, conversion_details, side, back = compute_overlap_metrics(side_data, back_data)

# 🔹 FN→TP 중 IDSW 발생한 경우 확인
fn_tp_with_idsw_count, fn_tp_with_idsw_details = compute_fn_tp_with_idsw(
    conversion_details, side_data, back_data
)


print("=== 중복된 IDSW, FP, FN 개수 (같은 프레임에서만) ===")
print(overlap_counts)
print("\n=== 전체 IDSW, FP, FN 개수 ===")
print(total_counts)
print("\n=== 중복도 (퍼센트) ===")
print(overlap_percentages)
print("\n=== FN에서 TP로 전환 가능 경우 개수 ===")
print(fn_tp_conversion_count)
print("\n=== 측면, FN에서 TP로 전환 가능 경우 개수 ===")
print(side)
print("\n=== 후면, FN에서 TP로 전환 가능 경우 개수 ===")
print(back)

# 필요시 상세 정보 출력:
# for frame, details in conversion_details.items():
#     print(f"🔹 프레임 {frame}: {details}")

# IDSW 보완 케이스 계산
total_complement, complement_details = compute_idsw_complement_metrics(side_data, back_data)

print("\n=== 보완 가능 ID Switch (IDSW) 케이스 총합 ===")
print(total_complement)
# print("\n=== 프레임별 보완된 ID Switch 상세 정보 ===")
# for frame, details in complement_details.items():
#     print(f"🔹 프레임 {frame}: {details}")

print("\n=== FN→TP 전환된 객체 중 IDSW 발생한 케이스 총합 ===")
print(fn_tp_with_idsw_count)

# 필요시 상세 정보 출력
for frame, details in fn_tp_with_idsw_details.items():
    print(f"🔹 프레임 {frame}: IDSW 포함된 FN→TP 전환 객체 {details}")

