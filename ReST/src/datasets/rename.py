import os
import shutil

# 📌 원본 이미지 폴더 설정
cam1_folder = "./BRT/sequence1/output/cam0/"
cam2_folder = "./BRT/sequence1/output/cam2/"
output_folder = "./BRT/sequence1/output/frames/"

# 📌 output 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 📌 파일 이름을 `{frame}_{camera}.jpg`로 변경하는 함수
def rename_and_move_images(cam_folder, cam_id):
    for file_name in sorted(os.listdir(cam_folder)):  # 파일 정렬
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # 지원하는 확장자 확인
            # 숫자만 추출 (프레임 번호)
            frame_number = ''.join(filter(str.isdigit, file_name))  

            # 새로운 파일명 지정
            new_file_name = f"{int(frame_number)}_{cam_id}.jpg"

            # 원본 파일 경로
            old_path = os.path.join(cam_folder, file_name)

            # 새로운 파일 경로
            new_path = os.path.join(output_folder, new_file_name)

            # 파일 이동 및 이름 변경
            shutil.move(old_path, new_path)
            print(f"✅ Moved: {old_path} → {new_path}")

# 📌 cam1 (0번 카메라) 이미지 변환
rename_and_move_images(cam1_folder, 0)

# 📌 cam2 (1번 카메라) 이미지 변환
rename_and_move_images(cam2_folder, 1)

print("🎯 모든 이미지가 정상적으로 변환되었습니다!")
