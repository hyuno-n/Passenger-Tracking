import cv2
import numpy as np

# 글로벌 변수
global points1, points2, points3, select_mode
points1, points2, points3 = [], [], []
select_mode = True  # 점 클릭 여부

# 현실 버스 배치도 좌표 (m 단위)
real_world_coords = [
    np.array([[330, 0], [330, 240], [555, 240], [555, 0]], dtype=np.float32), # 후면
    np.array([[255, 0], [255, 240], [405, 240], [405, 0]], dtype=np.float32), # 중앙
    np.array([[0, 0], [0, 240], [330, 240], [330, 0]], dtype=np.float32)    # 전면    
]

def add_padding(image, padding_x=150, padding_y=150, color=(0, 0, 0)):
    h, w = image.shape[:2]
    padded_image = cv2.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x,
                                      cv2.BORDER_CONSTANT, value=color)
    return padded_image, padding_x, padding_y

def draw_points_and_lines(image, points, color, alpha=0.5):
    if len(points) < 4:
        print("[WARNING] Not enough points to draw polygon.")
        return image
    try:
        points = np.array(points, dtype=np.int32)
        if len(points) >= 4:
            points = cv2.convexHull(points, returnPoints=True).reshape(-1, 2)
        for (x, y) in points:
            cv2.circle(image, (x, y), 5, color, -1)
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], color=color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.polylines(image, [points.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
    except Exception as e:
        print(f"[ERROR] draw_points_and_lines failed: {e}")
    return image

def click_event_image(event, x, y, flags, param):
    image_id, img, color = param
    global points1, points2, points3
    if event == cv2.EVENT_LBUTTONDOWN:
        if image_id == 1:
            points1.append((x, y))
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.imshow("Image 1 - Select Points", img)
            
        elif image_id == 2:
            points2.append((x, y))
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.imshow("Image 2 - Select Points", img)
            
        elif image_id == 3:
            points3.append((x, y))
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.imshow("Image 3 - Select Points", img)

# 예제 사전 좌표 (원하는 경우 수정 가능)
predefined_points1 = np.array([
    [1295, 651], [2034, 1221], [670, 685], [254, 1154]
], dtype=np.float32)

predefined_points2 = np.array([
    [498, 195], [753, 166], [97, 864], [1231, 761]
], dtype=np.float32)

predefined_points3 = np.array([
    [552.0,    211.0],   # 좌측 상단
    [858.0,    174.0],   # 좌측 하단
    [1372.0,   811.0],   # 우측 하단
    [86.0,     865.0]    # 우측 상단
], dtype=np.float32)  # 예시 값, 실제 이미지에 맞게 수정

# 이미지 로드
img1 = cv2.imread("view2_rear.jpg")
img2 = cv2.imread("view2_center.jpg")
img3 = cv2.imread("view2_front.jpg")  # 추가된 이미지

# 패딩 추가
img1_padded, _, _ = add_padding(img1, padding_x=500, padding_y=150)
img2_padded, _, _ = add_padding(img2, padding_x=300, padding_y=150)
img3_padded, _, _ = add_padding(img3, padding_x=300, padding_y=150)

# 선택 모드
if select_mode:
    # 이미지 1
    cv2.imshow("Image 1 - Select Points", img1_padded)
    cv2.setMouseCallback("Image 1 - Select Points", click_event_image, (1, img1_padded, (0, 0, 255)))
    while len(points1) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Image 1 - Select Points")
    print("Points1:", points1)

    # 이미지 2
    cv2.imshow("Image 2 - Select Points", img2_padded)
    cv2.setMouseCallback("Image 2 - Select Points", click_event_image, (2, img2_padded, (0, 255, 0)))
    while len(points2) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Image 2 - Select Points")

    # 이미지 3
    cv2.imshow("Image 3 - Select Points", img3_padded)
    cv2.setMouseCallback("Image 3 - Select Points", click_event_image, (3, img3_padded, (255, 0, 0)))
    while len(points3) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Image 3 - Select Points")
else:
    points1 = predefined_points1.tolist()
    points2 = predefined_points2.tolist()
    points3 = predefined_points3.tolist()

# Homography 계산
H1 = H2 = H3 = None
if len(points1) == 4:
    try:
        H1, _ = cv2.findHomography(np.array(points1, dtype=np.float32), real_world_coords[0], cv2.RANSAC)
        print("\n📌 Homography Matrix (Image 1):\n", H1)
        img1_padded = draw_points_and_lines(img1_padded, points1, (0, 0, 255))
    except Exception as e:
        print(f"[ERROR] Homography for Image 1 failed: {e}")

if len(points2) == 4:
    H2, _ = cv2.findHomography(np.array(points2, dtype=np.float32), real_world_coords[1], cv2.RANSAC)
    print("\n📌 Homography Matrix (Image 2):\n", H2)
    img2_padded = draw_points_and_lines(img2_padded, points2, (0, 255, 0))

if len(points3) == 4:
    H3, _ = cv2.findHomography(np.array(points3, dtype=np.float32), real_world_coords[2], cv2.RANSAC)
    print("\n📌 Homography Matrix (Image 3):\n", H3)
    img3_padded = draw_points_and_lines(img3_padded, points3, (255, 0, 0))

# 시각화
cv2.imshow("Image 1 Result", img1_padded)
cv2.imshow("Image 2 Result", img2_padded)
cv2.imshow("Image 3 Result", img3_padded)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
