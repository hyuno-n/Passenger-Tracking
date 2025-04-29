import os
import shutil
import random
from glob import glob

# 원본 데이터 폴더들
src_dirs = [
    "extracted_images_3frames",
]

# 결과 저장 경로
dst_root = "bus_dataset"
img_dst = os.path.join(dst_root, "images")
lbl_dst = os.path.join(dst_root, "labels")

for split in ["train", "val"]:
    os.makedirs(os.path.join(img_dst, split), exist_ok=True)
    os.makedirs(os.path.join(lbl_dst, split), exist_ok=True)

# 전체 이미지 수집
samples = []
for src in src_dirs:
    img_files = glob(os.path.join(src, "data", "*.*"))
    for img_path in img_files:
        base = os.path.basename(img_path)
        txt_path = os.path.join(src, "yolo", base.replace(".jpg", ".txt").replace(".png", ".txt"))
        if os.path.exists(txt_path):
            samples.append((img_path, txt_path))

# 셔플 및 분할
random.shuffle(samples)
split_ratio = 0.8
split_idx = int(len(samples) * split_ratio)
train_set = samples[:split_idx]
val_set = samples[split_idx:]

# 복사 함수
def copy_samples(sample_list, split):
    for img_path, txt_path in sample_list:
        base = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(img_dst, split, base))
        shutil.copy(txt_path, os.path.join(lbl_dst, split, base.replace(".jpg", ".txt").replace(".png", ".txt")))

# 실행
copy_samples(train_set, "train")
copy_samples(val_set, "val")

print(f"✅ Done! 총 이미지 수: {len(samples)} (Train: {len(train_set)}, Val: {len(val_set)})")

# YAML 파일 생성
yaml_path = os.path.join(dst_root, "head_data.yaml")
with open(yaml_path, "w") as f:
    f.write("path: dataset\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("\n")
    f.write("nc: 1\n")
    f.write("names: [\"head\"]\n")

print(f"📝 head_data.yaml 생성 완료 → {yaml_path}")
