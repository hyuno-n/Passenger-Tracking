import os
import shutil

# 시나리오별 프레임 범위 정의
scenarios = {
    "scen1": [(0, 33), (185, 300), (420, 562)],
    "scen2": [(317, 394)],
    "scen3": [(0, 58), (125, 225), (309, 406), (450, 562)],
    "scen4": [(0, 123), (216, 618), (698, 734), (784, 934)],
    "scen5": [(0, 68), (198, 362), (481, 507), (586, 660)],
    "scen6": [(132, 189)],
    "scen7": [(260, 371)],
    "scen8": [(0, 68), (234, 235)],
    "scen11": [(154, 314)],
    "scen12": [(0, 15), (258, 330), (378, 569)],
    "scen13": [(0, 44), (370, 572)],
    "scen14": [(0, 57), (257, 329)],
    "scen15": [(0, 56)]
}

# 유효한 이미지 확장자
valid_exts = ('.jpg', '.png', '.jpeg')

# 기준 디렉토리 (scen 폴더들이 있는 곳)
base_dir = "data/fisheye cam2"

# 출력 디렉토리
output_dir = "./data/scen_output"
os.makedirs(output_dir, exist_ok=True)

# 시나리오별 프레임 필터링 및 복사
for scen, ranges in scenarios.items():
    scen_dir = os.path.join(base_dir, scen)
    if not os.path.isdir(scen_dir):
        print(f"[스킵] 폴더 없음: {scen_dir}")
        continue

    # 출력 폴더 생성
    save_dir = os.path.join(output_dir, scen)
    os.makedirs(save_dir, exist_ok=True)

    # 이미지 파일 스캔
    for file in os.listdir(scen_dir):
        if not file.lower().endswith(valid_exts):
            continue

        # 파일명에서 숫자 추출 (예: frame_0123.jpg → 123)
        name, _ = os.path.splitext(file)
        digits = ''.join(filter(str.isdigit, name))
        if not digits:
            continue
        frame = int(digits)

        # 지정된 범위에 속하는지 확인
        in_range = any(start <= frame <= end for (start, end) in ranges)
        if in_range:
            src_path = os.path.join(scen_dir, file)
            dst_path = os.path.join(save_dir, file)
            shutil.copy2(src_path, dst_path)
            print(f"[복사] {file} → {scen}")

print(f"\n✅ 결과 저장 완료: {output_dir}")
