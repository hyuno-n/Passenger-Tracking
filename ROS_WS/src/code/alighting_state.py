import numpy as np
import cv2
import os
from ultralytics import YOLO
import sys
sys.path.append(os.path.abspath('/home/krri/Desktop/brt_bus/ros_ws/src/code/tools'))
from tools.hboe import hboe
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from PIL import Image as PILImage
from matplotlib.patches import Rectangle

def calculate_new_ori(ori):
    new_ori = []
    for o in ori:
        if o+90>=360:
            new_ori.append(o-270)
        else:
            new_ori.append(o+90)
    return new_ori

def draw_arrow(x, y, angle_deg, length, ax):
    angle_rad = np.deg2rad(angle_deg)
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    ax.arrow(x, y, dx, dy, head_width=0.07, head_length=0.07, fc='black', ec='black')

no_passenger = False
frame = None
vid_frame_count = 0
device="cpu"
view_img=True
exist_ok=False
# Setup Model
model = YOLO("yolov8x.pt")
model.to("cuda") if device == "0" else model.to("cpu")

# Extract classes names
names = model.names

# Video setup
videocapture = cv2.VideoCapture("/home/krri/Desktop/krri_ga/original_data/alight_and_board.mp4")
w, h, fps = (int(videocapture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
fig = plt.figure(figsize = (18, 6))

cnt = None
num = 0

ax1 = plt.subplot(1,3,1)
ax1_title = 'BEV alighting view: ' + str(round(videocapture.get(cv2.CAP_PROP_POS_MSEC)/1000, 1)) + ' sec'
ax1.set_title(ax1_title)
ax1.axis("off")

ax2 = plt.subplot(1, 3, 2)
ax2.set_title('alighting state')
ax2.set_xlim(0, 6)
ax2.set_ylim(-1, 5)

ax3 = plt.subplot(1, 3, 3)
ax3.set_title('predict alighting passengers')
ax3.axis("off")

# Rectangle 객체를 생성하고 Axes에 추가합니다.
reg_x1, reg_x2, reg_y1, reg_y2 = 150, 450, 515, 595
alighting_state = "Wait Alighting"
complete_time = 0
rect = Rectangle((reg_x1/100, 6-reg_y2/100), (reg_x2-reg_x1)/100, (reg_y2-reg_y1)/100, linewidth=1, edgecolor='g', facecolor='none')
ax2.add_patch(rect)

# Iterate over video frames
while videocapture.isOpened():
    
    def animate_door1(i):
        
        success, f = videocapture.read()
        
        if not success:
            return 1
        
        ax1_title = 'alighting view: ' + str(round(videocapture.get(cv2.CAP_PROP_POS_MSEC)/1000, 1)) + ' sec'
        ax1.set_title(ax1_title)
        ax3.set_title('predict alighting passengers')
    
        ##### 1. initial setting #####
        ax2.cla()
        ax2.set_xlim(0, 6)
        ax2.set_ylim(-1, 5)
        rect = Rectangle((reg_x1/100, 6-reg_y2/100), (reg_x2-reg_x1)/100, (reg_y2-reg_y1)/100, linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        
        # Run YOLOv8 inference on the frame
        results = model(f, classes=0, verbose=False)
        
        people_bottom = []
        people = []
        people_center = []
        people_ori = []
    
        eps = 1.2 # 반지름 (epsilon)
        
        in_region = False
        
        f2 = f.copy()

        boxes = results[0].boxes
        
        for box in boxes:
            b = box.xyxy[0]
            x1, y1, x2, y2 = b.tolist()
            cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            center_x = int((x1+x2)/2)
            center_y = int((y1+y2)/2)
            
            if center_y>reg_y2:
                continue
                
            crop_person = f[int(y1):int(y2), int(x1):int(x2)]
            person_ori = int(hboe(crop_person))
            
            if int((x1+x2)/2) >= 60 and int(y2) >= 125:
                people_center.append([center_x, center_y])
                people_ori.append(person_ori)
                
                people_bottom.append([[int((x1+x2)/2), int(y2)]])



        frame = f[0:600, 60:600]
        
        global alighting_state
        global cnt
        global complete_time
        global num
        global no_passenger
        
        if len(people_bottom) == 0:
            
            if no_passenger == False:
                
                complete_time = 0
                no_passenger = True
            
            alighting_state = "No Passenger"

            rows, cols = frame.shape[:2]

            black = np.zeros((rows*3, rows*3, 3), np.uint8)

            bg_height, bg_width, bg_channels = black.shape
            overlay_height, overlay_width, overlay_channels = frame.shape

            x_offset = (bg_width - overlay_width) // 2
            y_offset = (bg_height - overlay_height) // 2

            black[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = frame

            img = black        
            
            topLeft = [625, 725]
            bottomRight = [1800, 1200]  # x+y가 가장 큰 값이 우하단 좌표
            topRight = [1125, 725]  # x-y가 가장 작은 값이 우상단 좌표
            bottomLeft = [0, 1200]  # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = int(max([w1, w2]))  # 두 좌우 거리간의 최대값이 서류의 폭
            height = int(max([h1, h2]))  # 두 상하 거리간의 최대값이 서류의 높이

            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0],
                                [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            frame = cv2.warpPerspective(img, mtrx, (width, height))

            r, c = frame.shape[:2]


            frame = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_LINEAR)

            ##### 2. start alighting #####
            
            cv2.rectangle(frame, (reg_x1, reg_y1), (reg_x2, reg_y2), (0, 255, 0), 2)
            cv2.rectangle(f, (reg_x1, reg_y1), (reg_x2, reg_y2), (0, 255, 0), 2)
            
            ax2.set_title("No Passenger, Check Alighting Complete")

            complete_time += 1
            
            # print(complete_time)
            
            if complete_time%10 == 0:
                print("Alighting Complete Count: " + str(complete_time//10) + " sec")

            
            elif complete_time == 15: # fps = 10/1
                
                alighting_state = "Alighting Complete"    
                print("Alighting Complete Count: 1.5 sec, " + alighting_state)
                ax2.set_title(alighting_state)
                # print(alighting_state)
        
        else:
            no_passenger = False
            
            people_bottom = np.array(people_bottom)

            people_bottom[:, 0, 0] -= 60

            rows, cols = frame.shape[:2]

            black = np.zeros((rows*3, rows*3, 3), np.uint8)

            bg_height, bg_width, bg_channels = black.shape
            overlay_height, overlay_width, overlay_channels = frame.shape

            x_offset = (bg_width - overlay_width) // 2
            y_offset = (bg_height - overlay_height) // 2

            black[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = frame

            img = black        

            people_bottom[:, 0, 0] += 630
            people_bottom[:, 0, 1] += 600

            people_bottom = people_bottom.tolist()
            
            topLeft = [625, 725]
            bottomRight = [1800, 1200]  # x+y가 가장 큰 값이 우하단 좌표
            topRight = [1125, 725]  # x-y가 가장 작은 값이 우상단 좌표
            bottomLeft = [0, 1200]  # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = int(max([w1, w2]))  # 두 좌우 거리간의 최대값이 서류의 폭
            height = int(max([h1, h2]))  # 두 상하 거리간의 최대값이 서류의 높이

            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0],
                                [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            frame = cv2.warpPerspective(img, mtrx, (width, height))

            r, c = frame.shape[:2]

            transformed_people_bottom = []

            transformed_people_bottom = cv2.perspectiveTransform(np.float32(people_bottom), mtrx)

            frame = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_LINEAR)
            transformed_people_bottom[:, :, 1] *= 600/r
            transformed_people_bottom[:, :, 0] *= 600/c
            
            transformed_people_bottom = np.array(transformed_people_bottom[:, 0], dtype=int)
            transformed_people_bottom = transformed_people_bottom.tolist()

            for array in transformed_people_bottom:
                cv2.circle(frame, array, 4, (0, 255, 0), -1)

            ##### 2. start alighting #####
            
            cv2.rectangle(frame, (reg_x1, reg_y1), (reg_x2, reg_y2), (0, 255, 0), 2)
            cv2.rectangle(f, (reg_x1, reg_y1), (reg_x2, reg_y2), (0, 255, 0), 2)
            
            ##### 2. start alighting #####
            if alighting_state == "Wait Alighting":
                
                ax2.set_title(alighting_state)
                        
                for person_bottom, person_ori, person_center in zip(transformed_people_bottom, people_ori, people_center):

                    bottom_x, bottom_y = person_bottom
                    center_x, center_y = person_center
                    bx = bottom_x/100
                    by = 6-bottom_y/100
                    cx = center_x/100
                    cy = 6-center_y/100               
                    
                    p_ori = person_ori-270 if person_ori+90>=360 else person_ori+90

                    if reg_x1 < bottom_x < reg_x2 and reg_y1 < bottom_y < reg_y2:
                        
                        draw_arrow(bx, by, p_ori, eps, ax2)
                        
                        # if 100 < person_ori < 260:
                        if 100 < person_ori < 260:
                            in_region = True
                            
                            ax2.plot(bx, by, marker='o', color='red', linestyle='') # 그래프 초기화
                            # cv2.circle(frame, (bottom_x, bottom_y), 4, (0, 0, 255), -1)
                            cv2.circle(f2, (center_x, center_y), 4, (0, 0, 255), -1)

                            
                        else: 
                            ax2.plot(bx, by, marker='o', color='black', linestyle='') # 그래프 초기화
                            # cv2.circle(frame, (center_x, center_y), 4, (0, 0, 0), -1)
            
                    else:
                        ax2.plot(bx, by, marker='x', color='grey', linestyle='') # 그래프 초기화
                        # cv2.circle(frame, (center_x, center_y), 4, (0, 0, 0), -1)
                        
                if in_region == True:
            
                    alighting_state = "Start Alighting Passengers Recognition"
                    ax2.set_title(alighting_state)
                    print(alighting_state)
                                    
                        
            ##### 3. complete alighting #####            
            elif alighting_state == "Start Alighting Passengers Recognition":
            
                
                ax2.set_title("Alighting")
                
                if len(transformed_people_bottom)>0:
                
                    for i, p in enumerate(transformed_people_bottom):
                        
                        bottom_x, bottom_y = p
                        bx = bottom_x/100
                        by = 6-bottom_y/100

                        p_ori = people_ori[i]-270 if people_ori[i]+90>=360 else people_ori[i]+90
                        
                        if reg_x1 < bottom_x < reg_x2 and reg_y1 < bottom_y < reg_y2:
                            
                            draw_arrow(bx, by, p_ori, eps, ax2)
                            
                            # if 100 < people_ori[i] < 260:
                            if 100 < people_ori[i] < 260:
                                
                                in_region = True
                                
                                ax2.plot(bx, by, marker='o', color='red', linestyle='') # 그래프 초기화
                                # cv2.circle(frame, (bottom_x, bottom_y), 4, (0, 0, 255), -1)
                                cv2.circle(f2, (people_center[i][0], people_center[i][1]), 4, (0, 0, 255), -1)                        
                                
                            else: 
                                ax2.plot(bx, by, marker='o', color='black', linestyle='') # 그래프 초기화
                                # cv2.circle(frame, (bottom_x, bottom_y), 4, (0, 0, 0), -1)
                
                        else:
                            ax2.plot(bx, by, marker='x', color='grey', linestyle='') # 그래프 초기화
                            # cv2.circle(frame, (bottom_x, bottom_y), 4, (0, 0, 0), -1)
                            
                if in_region == True:
                    print("Keep Alighting, Reset Alighting Complete Count")
                    complete_time = 0
                    
                elif in_region == False:
                    
                    complete_time += 1
                    
                    # print(complete_time)
                    
                    if complete_time%10 == 0:
                        print("Alighting Complete Count: " + str(complete_time//10) + " sec")

                    
                    elif complete_time == 15: # fps = 10/1
                        print("Alighting Complete Count: 1.5 sec, " + alighting_state)
                        alighting_state = "Alighting Complete"    
                        ax2.set_title(alighting_state)
                        # print(alighting_state)
                                    

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame)
        ax3.imshow(f2)
        # print(alighting_state)
        # ax2.set_title(alighting_state)
        
        # Convert plot to image
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_np = image_np.reshape((height, width, 3))

        # Convert numpy array to PIL Image
        pil_image = PILImage.fromarray(image_np)
        
        path = "/home/krri/Desktop/krri_ga/"
        os.makedirs(path, exist_ok=True)
        # output_file = os.path.join('/home/ailleen/0429_brt/result/', f'img{num:03d}.jpg')
        cv2.imwrite(path + f'img{num:03d}.jpg', cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB))
        
        num += 1
        
        if alighting_state == "Alighting Complete":
            sys.exit()
    
    ani = FuncAnimation(plt.gcf(), animate_door1, interval=1000)
    
    plt.show()    

    plt.tight_layout()
    
    if animate_door1 == 1 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# 작업 완료 후 해제
videocapture.release()
cv2.waitKey(0)
cv2.destroyAllWindows()