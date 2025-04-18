import os
import cv2
import shutil
from tqdm import tqdm
import pathlib

def extract_frames(base_dir, target_frames, view_dirs, save_dir, frames_per_range=3):
    os.makedirs(save_dir, exist_ok=True)
    for scne, ranges in target_frames.items():
        for view in view_dirs:
            folder = os.path.join(base_dir, scne, "camera8_image_raw_flat_multi", view)
            if not os.path.isdir(folder):
                continue
            for r_start, r_end in ranges:
                if r_end - r_start < frames_per_range:
                    continue
                step = (r_end - r_start) // (frames_per_range + 1)
                selected_frames = [r_start + step * (i + 1) for i in range(frames_per_range)]
                for frame_idx in selected_frames:
                    filename = f"frame_{frame_idx:05d}.jpg"
                    file_path = os.path.join(folder, filename)
                    if not os.path.exists(file_path):
                        continue
                    save_name = f"{scne}_{view}_{filename}"
                    save_path = os.path.join(save_dir, save_name)
                    img = cv2.imread(file_path)
                    if img is not None:
                        cv2.imwrite(save_path, img)
    print("✅ 범위당 3장씩 이미지 추출 완료! 저장 폴더:", save_dir)

def separate_images_and_labels(src_dir, dst_img, dst_lbl):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    for file in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file)
        if file.endswith(".jpg"):
            shutil.copy(src_path, os.path.join(dst_img, file))
        elif file.endswith(".txt"):
            shutil.copy(src_path, os.path.join(dst_lbl, file))
    img_count = len(list(pathlib.Path(dst_img).glob("*.jpg")))
    txt_count = len(list(pathlib.Path(dst_lbl).glob("*.txt")))
    print(f"✅ 이미지: {img_count}장, 라벨: {txt_count}장")
    return img_count, txt_count

# ✅ 설정값들
target_frames = {
    "scne1": [(0, 81), (284, 306), (483, 590)],
    "scne1-1": [(378, 389)],
    "scne3": [(0, 50), (208, 224)],
    "scne4": [(0, 176), (403, 780), (856, 1066)],
    "scne5": [(0, 61), (341, 426), (547, 800)],
    "scne7": [(314, 373)],
    "scne8": [(0, 113), (183, 229)],
    "scne9": [(0, 73)],
    "scne11": [(107, 134), (236, 362)],
    "scne12": [(0, 20), (370, 605)],
    "scne13": [(0, 45), (512, 558)],
    "scne14": [(0, 150), (336, 376)],
    "scne15": [(0, 158)],
}

view_dirs = ["view_-40", "view_0", "view_40"]
base_dir = "/media/krri/Samsung_T54/BRT_data/ex1"
save_dir = "extracted_images_3frames"
dst_img = "bus_dataset/images/train"
dst_lbl = "bus_dataset/labels/train"

# ✅ 실행
extract_frames(base_dir, target_frames, view_dirs, save_dir)
separate_images_and_labels(save_dir, dst_img, dst_lbl)
