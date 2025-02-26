import json
import numpy as np

# JSON 파일 로드 함수
def load_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 객체 ID별 이벤트 추출 함수
def extract_event_by_object(metrics):
    idsw_objects = metrics["ID Switch Objects"]  # {object_id: 발생 프레임 리스트}
    fp_objects = metrics["False Positive Objects"]  # {object_id: 발생 프레임 리스트}
    fn_objects = metrics["False Negative Objects"]  # {object_id: 발생 프레임 리스트}
    return idsw_objects, fp_objects, fn_objects

# Jaccard Index 계산 함수 (객체별 동일 이벤트 발생 여부 체크)
def jaccard_index(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# JSON 파일 불러오기
front_metrics = load_metrics("front_metrics.json")
side_metrics = load_metrics("side_metrics.json")
back_metrics = load_metrics("back_metrics.json")

# 이벤트 객체별 추출
front_idsw, front_fp, front_fn = extract_event_by_object(front_metrics)
side_idsw, side_fp, side_fn = extract_event_by_object(side_metrics)
back_idsw, back_fp, back_fn = extract_event_by_object(back_metrics)

# 객체 ID 리스트 추출 (모든 시점의 객체 ID 집합)
all_object_ids = set(front_idsw.keys()) | set(side_idsw.keys()) | set(back_idsw.keys())

# 객체별 중복 이벤트 체크
print("\n🔍 동일 객체에서 IDSW, FP, FN 발생 중복 검사")
for obj_id in all_object_ids:
    idsw_set = set(front_idsw.get(obj_id, [])) | set(side_idsw.get(obj_id, [])) | set(back_idsw.get(obj_id, []))
    fp_set = set(front_fp.get(obj_id, [])) | set(side_fp.get(obj_id, [])) | set(back_fp.get(obj_id, []))
    fn_set = set(front_fn.get(obj_id, [])) | set(side_fn.get(obj_id, [])) | set(back_fn.get(obj_id, []))

    idsw_fp_overlap = jaccard_index(idsw_set, fp_set)
    idsw_fn_overlap = jaccard_index(idsw_set, fn_set)
    fp_fn_overlap = jaccard_index(fp_set, fn_set)

    print(f"🔹 객체 {obj_id}:")
    print(f"   - IDSW & FP 중복도: {idsw_fp_overlap:.3f}")
    print(f"   - IDSW & FN 중복도: {idsw_fn_overlap:.3f}")
    print(f"   - FP & FN 중복도: {fp_fn_overlap:.3f}")
