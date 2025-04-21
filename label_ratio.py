import os
import json
from collections import defaultdict

# 설정
label_dir = "./data/scen_label"  # scen1.json, scen2.json, ...
seat_ids = [f"S{i}" for i in range(1, 16)]  # S1 ~ S15

# 결과 저장용
seat_counts = {sid: {"True": 0, "False": 0} for sid in seat_ids}

# 전체 시나리오 루프
for filename in sorted(os.listdir(label_dir)):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(label_dir, filename)
    with open(path, "r") as f:
        label_data = json.load(f)

    for frame in label_data.values():
        for sid in seat_ids:
            val = frame.get(sid, False)  # 값이 없으면 False 처리
            seat_counts[sid][str(val)] += 1

# 비율 출력
print("📊 좌석별 True/False 비율 (%):")
for sid in seat_ids:
    t = seat_counts[sid]["True"]
    f = seat_counts[sid]["False"]
    total = t + f
    true_ratio = (t / total) * 100 if total > 0 else 0
    false_ratio = (f / total) * 100 if total > 0 else 0
    print(f"{sid}: True {true_ratio:.1f}% | False {false_ratio:.1f}% (총 {total} 프레임)")
