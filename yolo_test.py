import cv2
import numpy as np
from scipy.ndimage import map_coordinates

# ------------------------
# 1. 사각형 내부 그리드 생성
# ------------------------
def generate_grid_points(rect, rows=10, cols=10):
    x1, y1 = rect[0]
    x2, y2 = rect[2]
    grid = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            u = x1 + (x2 - x1) * j / cols
            v = y1 + (y2 - y1) * i / rows
            grid.append([u, v])
    return np.array(grid, dtype=np.float32).reshape((rows + 1, cols + 1, 2))

# ------------------------
# 2. 좌표 맵 생성 (어안 → 평면 매핑 좌표 기록)
# ------------------------
def build_fisheye_to_plane_map(ih, iw, o_sz=800, i_fov=190, o_fov=120, o_u=0, o_v=0):
    i_fov = np.deg2rad(i_fov)
    o_fov = np.deg2rad(o_fov)
    o_u = np.deg2rad(o_u)
    o_v = np.deg2rad(o_v)

    z = 1
    L = np.tan(o_fov / 2) / z
    x = np.linspace(L, -L, num=o_sz)
    y = np.linspace(-L, L, num=o_sz)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.ones_like(x_grid)

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

    Rx = get_rotation_matrix(o_v, [1, 0, 0])
    Ry = get_rotation_matrix(o_u, [0, 1, 0])
    xyz = np.stack([x_grid, y_grid, z_grid], -1).dot(Rx).dot(Ry)

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    theta = np.arctan2(y, x)
    c = np.sqrt(x**2 + y**2)
    rho = np.arctan2(c, z)
    r = rho * min(ih, iw) / i_fov

    coor_x = r * np.cos(theta) + iw / 2
    coor_y = r * np.sin(theta) + ih / 2

    return coor_x.astype(np.float32), coor_y.astype(np.float32)

# ------------------------
# 3. 곡선 사각형 시각화 (맵 기반 역보정)
# ------------------------
def draw_rect_from_plane_grid(fisheye_img, plane_rect, map_x, map_y, o_sz=800):
    ih, iw = fisheye_img.shape[:2]
    rows, cols = 10, 10
    grid_pts = generate_grid_points(plane_rect, rows, cols)

    result = fisheye_img.copy()
    for i in range(rows + 1):
        for j in range(cols + 1):
            px = int(np.clip(grid_pts[i, j, 0], 0, o_sz - 1))
            py = int(np.clip(grid_pts[i, j, 1], 0, o_sz - 1))
            x = int(map_x[py, px])
            y = int(map_y[py, px])
            cv2.circle(result, (x, y), 1, (0, 255, 0), -1)

    return result

# ------------------------
# 4. 좌석 사각형 정의
# ------------------------
def define_seat_rects(seat_width=75, seat_height=50, seat_start_x=30, seat_start_y=0):
    seat_rects = []
    for row in range(2):
        num_seats = 5 if row == 1 else 7
        for col in range(num_seats):
            x = seat_start_x + col * seat_width
            y = seat_start_y + row * seat_height
            rect_pts = np.array([
                [x, y],
                [x + seat_width, y],
                [x + seat_width, y + seat_height],
                [x, y + seat_height]
            ], dtype=np.float32)
            seat_rects.append(rect_pts)

    extra_y = 190
    extra_x1 = seat_start_x + 2 * seat_width
    extra_x2 = seat_start_x + 3 * seat_width
    seat_rects.append(np.array([
        [extra_x1, extra_y],
        [extra_x1 + seat_width, extra_y],
        [extra_x1 + seat_width, extra_y + seat_height],
        [extra_x1, extra_y + seat_height]
    ], dtype=np.float32))
    seat_rects.append(np.array([
        [extra_x2, extra_y],
        [extra_x2 + seat_width, extra_y],
        [extra_x2 + seat_width, extra_y + seat_height],
        [extra_x2, extra_y + seat_height]
    ], dtype=np.float32))

    return seat_rects

# ------------------------
# 5. 실행부
# ------------------------
if __name__ == "__main__":
    image_path = "test.jpg"
    fisheye_img = cv2.imread(image_path)
    ih, iw = fisheye_img.shape[:2]

    # 좌표 맵 생성
    map_x, map_y = build_fisheye_to_plane_map(ih, iw, o_sz=800)

    # 평면 기준 좌석들 정의
    seat_rects = define_seat_rects()

    result = fisheye_img.copy()
    for rect in seat_rects:
        result = draw_rect_from_plane_grid(result, rect, map_x, map_y)

    cv2.imshow("Fisheye Seat Mapping (Inverse)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
