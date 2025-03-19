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
    [330, 0],    # 우측 하단
    [330, 330]   # 우측 상단
], dtype=np.float32)

def add_padding(image, padding=150, color=(0, 0, 0)):
    """이미지 상하좌우에 패딩 추가"""
    h, w = image.shape[:2]
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                      cv2.BORDER_CONSTANT, value=color)
    return padded_image, padding

def click_event_image1(event, x, y, flags, param):
    global points1, select_mode
    if select_mode and event == cv2.EVENT_LBUTTONDOWN:
        points1.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Padded Image 1 - Select Points", param)
        if len(points1) == 4:
            cv2.destroyWindow("Padded Image 1 - Select Points")
            print("\n📌 선택된 좌표 (Image 1):", points1)

def click_event_image2(event, x, y, flags, param):
    global points2, select_mode
    if select_mode and event == cv2.EVENT_LBUTTONDOWN:
        points2.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Padded Image 2 - Select Points", param)
        if len(points2) == 4:
            cv2.destroyWindow("Padded Image 2 - Select Points")
            print("\n📌 선택된 좌표 (Image 2):", points2)

def toggle_selection():
    """선택 모드를 활성화/비활성화"""
    global select_mode
    select_mode = not select_mode
    print("\n📌 선택 모드:", "활성화" if select_mode else "비활성화")

# 기존 이미지별 4개 좌표 (예제 좌표, 필요시 수정 가능)
predefined_points1 = np.array([
    [1029, 679], [1623, 1135], [473, 707], [191, 1127]
], dtype=np.float32)

predefined_points2 = np.array([
    [335, 230], [598, 197], [31, 703], [935, 699]
], dtype=np.float32)

# 이미지 로드
img1 = cv2.imread("image1.jpg")  # Cam1 (측면)
img2 = cv2.imread("image2.jpg")  # Cam2-1 (후면)

# 패딩 추가
img1_padded, padding1 = add_padding(img1, padding=500)  # 500px 패딩 추가
img2_padded, padding2 = add_padding(img2, padding=150)

# 선택 모드 활성화 시 클릭 이벤트 실행
if select_mode:
    # 첫 번째 이미지에서 4개 점 선택
    cv2.imshow("Padded Image 1 - Select Points", img1_padded)
    cv2.setMouseCallback("Padded Image 1 - Select Points", click_event_image1, img1_padded.copy())
    cv2.waitKey(0)

    # 두 번째 이미지에서 4개 점 선택
    cv2.imshow("Padded Image 2 - Select Points", img2_padded)
    cv2.setMouseCallback("Padded Image 2 - Select Points", click_event_image2, img2_padded.copy())
    cv2.waitKey(0)

cv2.destroyAllWindows()

# 선택 모드가 비활성화된 경우, 미리 정의된 좌표 사용
if not select_mode:
    points1 = predefined_points1.tolist()
    points2 = predefined_points2.tolist()

# Homography 행렬 계산
if len(points1) == 4:
    src_pts1 = np.array(points1, dtype=np.float32)
    H1, _ = cv2.findHomography(src_pts1, real_world_coords, cv2.RANSAC)
    print("\n📌 계산된 Homography 행렬 (Image 1):\n", H1)

if len(points2) == 4:
    src_pts2 = np.array(points2, dtype=np.float32)
    H2, _ = cv2.findHomography(src_pts2, real_world_coords, cv2.RANSAC)
    print("\n📌 계산된 Homography 행렬 (Image 2):\n", H2)
