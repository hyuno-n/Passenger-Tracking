import os
import numpy as np
import cv2
import json
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

class BusDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.num_cam = 2  # 후면 & 측면 카메라 2개
        self.img_shape = [1080, 1920]  # 이미지 크기
        self.num_frame = 1200  # 총 프레임 수
        self.frames = sorted([f for f in os.listdir(os.path.join(root, "Image_subsets/rear_camera")) if f.endswith(".jpg")])

        # Load calibration files
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        cam_names = ["rear", "side"]
        intrinsic_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero', f"intr_{cam_names[camera_i]}.xml")
        extrinsic_path = os.path.join(self.root, 'calibrations', 'extrinsic', f"extr_{cam_names[camera_i]}.xml")

        intrinsic_file = cv2.FileStorage(intrinsic_path, flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_file.getNode('camera_matrix').mat()
        intrinsic_file.release()

        extrinsic_root = ET.parse(extrinsic_path).getroot()
        rvec = np.array(list(map(float, extrinsic_root.findall('rvec')[0].text.split())), dtype=np.float32)
        tvec = np.array(list(map(float, extrinsic_root.findall('tvec')[0].text.split())), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec.reshape(3, 1)))

        return intrinsic_matrix, extrinsic_matrix

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame_name = self.frames[index]
        rear_img = cv2.imread(os.path.join(self.root, "Image_subsets/rear_camera", frame_name))
        side_img = cv2.imread(os.path.join(self.root, "Image_subsets/side_camera", frame_name))

        label_path = os.path.join(self.root, "labels", frame_name.replace(".jpg", ".txt"))
        with open(label_path, "r") as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]

        return {
            "rear_img": rear_img,
            "side_img": side_img,
            "labels": np.array(labels),
            "frame": frame_name
        }
