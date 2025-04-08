import cv2
import numpy as np

# 전체 좌석 리스트
all_seat_points = []
current_seat = []

# 좌석별 색상 리스트
seat_colors = [
    (0, 255, 0),     # green
    (0, 255, 255),   # yellow
    (255, 0, 0),     # blue
    (255, 0, 255),   # magenta
    (0, 128, 255),   # orange
    (128, 0, 255),   # purple
    (255, 128, 0),   # light orange
    (128, 255, 0),   # lime
]

def draw_seat(img, points, seat_idx):
    """선택된 좌표로 사각형을 그리고 시각화 (좌석별 색상 지정)"""
    color = seat_colors[seat_idx % len(seat_colors)]  # 좌석 개수 초과 시 반복
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color=color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    # 좌석 번호 표시
    cx, cy = np.mean(pts.reshape(4, 2), axis=0).astype(int)
    cv2.putText(img, f"{seat_idx+1}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def mouse_callback(event, x, y, flags, param):
    global current_seat
    img = param['img']
    display = param['display']

    if event == cv2.EVENT_LBUTTONDOWN and len(current_seat) < 4:
        current_seat.append((x, y))
        print(f"📍 점 {len(current_seat)}: ({x}, {y})")
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

        if len(current_seat) == 4:
            draw_seat(display, current_seat, len(all_seat_points))  # 좌석 색상 적용
            cv2.imshow("Select Multiple Seats", display)

# 이미지 로드
img = cv2.imread("image1.jpg")
if img is None:
    raise FileNotFoundError("❌ 이미지를 불러올 수 없습니다.")

display_img = img.copy()
cv2.imshow("Select Multiple Seats", display_img)
cv2.setMouseCallback("Select Multiple Seats", mouse_callback, {'img': img, 'display': display_img})

print("🪑 좌석 하나당 4개의 점을 클릭해 정의하세요.")
print("↩️ Enter를 누르면 다음 좌석 정의로 넘어갑니다.")
print("❌ 'q' 키를 누르면 종료됩니다.")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Enter 키 → 다음 좌석
        if len(current_seat) == 4:
            all_seat_points.append(current_seat.copy())
            print(f"✅ 좌석 {len(all_seat_points)} 저장 완료.\n")
            current_seat.clear()
        else:
            print("⚠️ 4개의 점을 모두 찍어야 저장됩니다.")

    elif key == ord('q'):  # 종료
        break

cv2.destroyAllWindows()

print("\n📦 최종 좌석 좌표 목록:")
for idx, seat in enumerate(all_seat_points):
    print(f"Seat {idx + 1}: {seat}")
