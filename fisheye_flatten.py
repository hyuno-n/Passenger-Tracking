import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import map_coordinates

# ========= Rotation Matrix ==========
def get_rotation_matrix(rad, ax):
    ax = np.array(ax)
    ax = ax / np.linalg.norm(ax)
    R = np.diag([np.cos(rad)] * 3)
    R += np.outer(ax, ax) * (1.0 - np.cos(rad))
    ax *= np.sin(rad)
    R += np.array([[0, -ax[2], ax[1]],
                   [ax[2], 0, -ax[0]],
                   [-ax[1], ax[0], 0]])
    return R

# ========= Grid Projection ==========
def grid_in_3d_to_project(o_fov, o_sz, o_u, o_v):
    z = 1
    L = np.tan(o_fov / 2) / z
    x = np.linspace(L, -L, num=o_sz)
    y = np.linspace(-L, L, num=o_sz)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.ones_like(x_grid)
    Rx = get_rotation_matrix(o_v, [1, 0, 0])
    Ry = get_rotation_matrix(o_u, [0, 1, 0])
    xyz = np.stack([x_grid, y_grid, z_grid], -1).dot(Rx).dot(Ry)
    return [xyz[..., i] for i in range(3)]

# ========= 단일 뷰 평면화 ==========
def fisheye_to_plane(frame, ih, iw, i_fov=180, o_fov=120, o_sz=800, o_u=0, o_v=0):
    i_fov = np.deg2rad(i_fov)
    o_fov = np.deg2rad(o_fov)
    o_u = np.deg2rad(o_u)
    o_v = np.deg2rad(o_v)
    x_grid, y_grid, z_grid = grid_in_3d_to_project(o_fov, o_sz, o_u, o_v)
    theta = np.arctan2(y_grid, x_grid)
    c = np.sqrt(x_grid**2 + y_grid**2)
    rho = np.arctan2(c, z_grid)
    r = rho * min(ih, iw) / i_fov
    coor_x = r * np.cos(theta) + iw / 2
    coor_y = r * np.sin(theta) + ih / 2
    coor_x = np.clip(coor_x, 0, iw - 1)
    coor_y = np.clip(coor_y, 0, ih - 1)
    flat = np.stack([
        map_coordinates(frame[..., ch], [coor_y, coor_x], order=1)
        for ch in range(frame.shape[-1])
    ], axis=-1)
    return np.fliplr(flat)

def make_square_with_padding(img):
    h, w = img.shape[:2]
    size = max(h, w)

    # top, bottom, left, right 패딩 계산
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left

    # 검정색으로 패딩 추가
    squared = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # 검정색
    )
    return squared

# ========= main multi-view flattening ==========
def apply_multi_view_flattening_flatfolder(base_dir):
    view_angles = [-40, 0, 40]

    for folder in sorted(os.listdir(base_dir)):
        if not folder.startswith("scen2"):
            continue

        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
        if not img_files:
            print(f"❌ [스킵] '{folder_path}' 이미지 없음")
            continue

        print(f"\n📷 [MULTI-FLAT] 처리 중: {folder}")

        for i, filename in enumerate(tqdm(img_files, desc=folder)):
            img_path = os.path.join(folder_path, filename)
            try:
                frame = cv2.imread(img_path)

                # ⬇️ 정사각형 padding 적용
                frame = make_square_with_padding(frame)

                ih, iw = frame.shape[:2]

                for angle in view_angles:
                    flat = fisheye_to_plane(
                        frame, ih, iw,
                        o_fov=95, o_sz=800,
                        o_u=angle, o_v=0
                    )

                    # 회전 처리
                    if angle == -40:
                        flat = cv2.rotate(flat, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif angle == 40:
                        flat = cv2.rotate(flat, cv2.ROTATE_90_CLOCKWISE)

                    view_dir = os.path.join(folder_path, f"view_{angle}")
                    os.makedirs(view_dir, exist_ok=True)
                    save_path = os.path.join(view_dir, filename)
                    cv2.imwrite(save_path, flat)

            except Exception as e:
                print(f"❌ 처리 실패: {filename} - {e}")
                continue

    print("\n🎉 모든 폴더 이미지 multi-view 평면화 완료")

# ========= CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="scen_output 폴더 경로")
    args = parser.parse_args()

    apply_multi_view_flattening_flatfolder(args.base_dir)
