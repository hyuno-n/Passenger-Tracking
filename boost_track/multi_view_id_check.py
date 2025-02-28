import json

def load_json(file_path):
    """ JSON 파일을 로드하는 함수 """
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_overlap_metrics(side_results, back_results):
    """ 
    두 JSON 데이터에서 같은 프레임 ID에서만 IDSW, FP, FN의 중복도를 계산하는 함수 
    """
    overlap_counts = {"ID Switches": 0, "False Positives": 0, "False Negatives": 0}
    total_counts = {"ID Switches": 0, "False Positives": 0, "False Negatives": 0}
    overlapping_details = {}  # 중복된 프레임 ID에서 발생한 객체 정보 저장

    common_frames = set(side_results.keys()).intersection(set(back_results.keys()))  # 공통된 프레임만 비교

    for frame in common_frames:
        side_data = side_results[frame]
        back_data = back_results[frame]

        frame_overlap = {}  # 해당 프레임의 중복된 IDSW, FP, FN을 저장할 딕셔너리

        for key in ["ID Switches", "False Positives", "False Negatives"]:
            side_ids = set(side_data[key])
            back_ids = set(back_data[key])

            # 전체 개수 기록
            total_counts[key] += len(side_ids) + len(back_ids)

            # 중복된 개수 계산
            common_ids = side_ids.intersection(back_ids)
            overlap_counts[key] += len(common_ids)

            if common_ids:  # 중복된 값이 있으면 기록
                frame_overlap[key] = list(common_ids)

        if frame_overlap:  # 중복된 프레임 정보가 있으면 저장
            overlapping_details[frame] = frame_overlap

    # 중복도 계산 (퍼센트)
    overlap_percentages = {
        key: (overlap_counts[key] / total_counts[key] * 100 if total_counts[key] > 0 else 0)
        for key in overlap_counts
    }

    return overlap_counts, total_counts, overlap_percentages, overlapping_details

# 파일 경로 설정
side_file_path = "./tracking_results/side_tracking_results.json"
back_file_path = "./tracking_results/back_tracking_results.json"

# JSON 데이터 로드
side_data = load_json(side_file_path)
back_data = load_json(back_file_path)

# 중복도 계산
overlap_counts, total_counts, overlap_percentages, overlapping_details = compute_overlap_metrics(side_data, back_data)

# 결과 출력
print("=== 중복된 IDSW, FP, FN 개수 (같은 프레임에서만) ===")
print(overlap_counts)

print("\n=== 전체 IDSW, FP, FN 개수 ===")
print(total_counts)

print("\n=== 중복도 (퍼센트) ===")
print(overlap_percentages)

print("\n=== 중복된 프레임 상세 정보 ===")
for frame, details in overlapping_details.items():
    print(f"🔹 프레임 {frame}:")
    for key, ids in details.items():
        print(f"  - {key}: {ids}")
