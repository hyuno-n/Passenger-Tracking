import cv2
import numpy as np

# 글로벌 변수 설정
global points1, points2, select_mode
points1 = []
points2 = []
select_mode = True  # 선택 모드 활성화 여부

# 현실 버스 배치도에서의 대응 좌표 (미터 단위)
real_world_coords = np.array([
    [0, 0],      # 좌측 하단
    [0, 240],    # 좌측 상단
    [550, 0],    # 우측 하단
    [550, 240]   # 우측 상단
], dtype=np.float32)

def add_padding(image, padding_x=150, padding_y=150, color=(0, 0, 0)):
    h, w = image.shape[:2]
    padded_image = cv2.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x,
                                      cv2.BORDER_CONSTANT, value=color)
    return padded_image, padding_x, padding_y

def draw_points_and_lines(image, points, color, alpha=0.5):
    points = np.array(points, dtype=np.int32)
    if len(points) >= 4:
        points = cv2.convexHull(points, returnPoints=True)
        points = points.reshape(-1, 2)
    for (x, y) in points:
        cv2.circle(image, (x, y), 5, color, -1)
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color=color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.polylines(image, [points.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
    return image

def click_event_image(event, x, y, flags, param):
    image_id, img, color = param
    global points1, points2
    if event == cv2.EVENT_LBUTTONDOWN:
        if image_id == 1:
            points1.append((x, y))
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.imshow("Padded Image 1 - Select Points", img)
            if len(points1) == 4:
                cv2.destroyWindow("Padded Image 1 - Select Points")
        elif image_id == 2:
            points2.append((x, y))
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.imshow("Padded Image 2 - Select Points", img)
            if len(points2) == 4:
                cv2.destroyWindow("Padded Image 2 - Select Points")

# 기존 이미지별 4개 좌표 (예제 좌표, 필요시 수정 가능)
predefined_points1 = np.array([
    [1295, 651], [2034, 1221], [670, 685], [254, 1154]
], dtype=np.float32)

predefined_points2 = np.array([
    [498, 195], [753, 166], [97, 864], [1231, 761]
], dtype=np.float32)

# 이미지 로드
img1 = cv2.imread("front_frame_00348.jpg")
img2 = cv2.imread("back_frame_00348.jpg")

# 패딩 추가
img1_padded, padding1_x, padding1_y = add_padding(img1, padding_x=300, padding_y=150)
img2_padded, padding2_x, padding2_y = add_padding(img2, padding_x=300, padding_y=150)

# 선택 모드
if select_mode:
    cv2.imshow("Padded Image 1 - Select Points", img1_padded)
    cv2.setMouseCallback("Padded Image 1 - Select Points", click_event_image, (1, img1_padded.copy(), (0, 0, 255)))
    cv2.waitKey(0)
    
    cv2.imshow("Padded Image 2 - Select Points", img2_padded)
    cv2.setMouseCallback("Padded Image 2 - Select Points", click_event_image, (2, img2_padded.copy(), (0, 255, 0)))
    cv2.waitKey(0)
else:
    points1 = predefined_points1.tolist()
    points2 = predefined_points2.tolist()

# Homography 행렬 계산
H1, H2 = None, None
if len(points1) == 4:
    src_pts1 = np.array(points1, dtype=np.float32)
    H1, _ = cv2.findHomography(src_pts1, real_world_coords, cv2.RANSAC)
    print("\n📌 계산된 Homography 행렬 (Image 1):\n", H1)
    img1_padded = draw_points_and_lines(img1_padded, points1, (0, 0, 255))

if len(points2) == 4:
    src_pts2 = np.array(points2, dtype=np.float32)
    H2, _ = cv2.findHomography(src_pts2, real_world_coords, cv2.RANSAC)
    print("\n📌 계산된 Homography 행렬 (Image 2):\n", H2)
    img2_padded = draw_points_and_lines(img2_padded, points2, (0, 255, 0))

cv2.imshow("Padded Image 1 with Points and Plane", img1_padded)
cv2.imshow("Padded Image 2 with Points and Plane", img2_padded)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
