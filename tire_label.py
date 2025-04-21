import os

def append_labels_to_view40(folder):
    # 추가할 라벨 정보
    new_labels = [
        "1 0.081250 0.411875 0.097500 0.111250",
        "1 0.268125 0.223750 0.028750 0.052500"
    ]

    count = 0
    for file in os.listdir(folder):
        if file.endswith(".txt") and "view_40" in file:
            label_name = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(folder, label_name)

            # append 두 줄
            with open(label_path, "a") as f:
                for line in new_labels:
                    f.write(line + "\n")
            count += 1

    print(f"✅ 총 {count}개의 view_40 라벨 파일에 라벨을 추가했습니다.")

# 예시 사용
folder_path = "bus_dataset/labels/train"  # 🔁 여기에 라벨/이미지 파일들이 있는 폴더 경로를 입력하세요
append_labels_to_view40(folder_path)
