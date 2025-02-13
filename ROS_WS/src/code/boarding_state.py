import numpy as np
import cv2
import os
from ultralytics import YOLO
import sys
sys.path.append(os.path.abspath('/home/krri/Desktop/krri_ga/code/tools'))
from tools.hboe import hboe
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from PIL import Image as PILImage
from matplotlib.patches import Rectangle

def draw_arrow(x, y, angle_deg, length, ax):
    angle_rad = np.deg2rad(angle_deg)
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    ax.arrow(x, y, dx, dy, head_width=0.07, head_length=0.07, fc='black', ec='black')

def calculate_new_ori(ori):
    new_ori = []
    for o in ori:
        if o+90>=360:
            new_ori.append(o-270)
        else:
            new_ori.append(o+90)
    return new_ori

def calculate_new_data(data, angle_deg, length):
    new_data = []
    for d in range(len(data)):
        angle_rad = np.deg2rad(angle_deg[d])
        end = (data[d][0] + length * np.cos(angle_rad), data[d][1] + length * np.sin(angle_rad))
        new_data.append(end)
    return new_data

def calculate_new_data_reverse(data, angle_deg, length):
    if angle_deg+180>360:
        ori = angle_deg-180
    else:
        ori = angle_deg+180
    angle_rad = np.deg2rad(ori)
    end = (data[0] + length * np.cos(angle_rad), data[1] + length * np.sin(angle_rad))

    return end

def draw_eps(x, y, eps, colors, ax):
    # 플롯 생성 (마커는 'o'로 설정하고, 투명도는 0.5, 원의 반지름은 eps으로 설정)
    ax.plot(x, y, marker='o', linestyle='', alpha=0.15, color=colors, markersize=eps*53)


def find_root(roots, i):
    while roots[i] != i:
        i = roots[i]
    return i

def union(roots, ranks, i, j):
    root_i = find_root(roots, i)
    root_j = find_root(roots, j)
    if root_i != root_j:
        if ranks[root_i] > ranks[root_j]:
            roots[root_j] = root_i
        elif ranks[root_i] < ranks[root_j]:
            roots[root_i] = root_j
        else:
            roots[root_j] = root_i
            ranks[root_i] += 1

def merge_lists(lists):
    n = len(lists)
    roots = list(range(n))
    ranks = [0] * n
    element_to_root = {}  # Dictionary to map element to its root index
    
    for i, lst in enumerate(lists):
        for elem in lst:
            elem_tuple = tuple(elem)  # Convert element to tuple to use as dictionary key
            if elem_tuple in element_to_root:
                union(roots, ranks, i, element_to_root[elem_tuple])
            element_to_root[elem_tuple] = find_root(roots, i)
    
    index_to_elements = {}
    for i, lst in enumerate(lists):
        root = find_root(roots, i)
        if root not in index_to_elements:
            index_to_elements[root] = set()
        index_to_elements[root].update(tuple(e) for e in lst)
    
    result = [sorted(list(elements)) for elements in index_to_elements.values() if elements]
    return result

def Euclidean(v1, v2): # 유클리드 거리 계산 함수
    sum = 0
    for i in range(len(v1)):
        sum +=  (v1[i]-v2[i])**2
    return round(math.sqrt(sum), 2)

def cluster_new(data, points, eps):
    
    cluster = []
    
    for p in points:

        c = []
        for d in data:
            
            if points.index(p) != data.index(d) :
            
                dist = Euclidean(d, p)
                if dist < eps:
                    c.append(d)
            
            else:
                
                c.append(d)
                
        cluster.append(c)
        
    # print(cluster)
        
    cluster = merge_lists(cluster)
    return cluster


frame = None
vid_frame_count = 0
view_img=True
exist_ok=False
# Setup Model
model = YOLO("yolov8x.pt")
model.to("cuda")

# Extract classes names
names = model.names

# Video setup
videocapture = cv2.VideoCapture("/home/krri/Desktop/krri_ga/original_data/alight_and_board.mp4")

w, h, fps = (int(videocapture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

fig = plt.figure(figsize = (18, 6))
cnt = None
num = 0
queue_cnt = 0
start_cnt = 0

ax1 = plt.subplot(1,3,1)
ax1_title = 'BEV boarding view: ' + str(round(videocapture.get(cv2.CAP_PROP_POS_MSEC)/1000, 1)) + ' sec'
ax1.set_title(ax1_title)
ax1.axis("off")

ax2 = plt.subplot(1, 3, 2)
ax2.set_title('boarding state')
ax2.set_xlim(0, 6)
ax2.set_ylim(-1, 5)

ax3 = plt.subplot(1, 3, 3)
ax3.set_title('predict boarding passengers')
ax3.axis("off")

# Rectangle 객체를 생성하고 Axes에 추가합니다.
reg_x1, reg_x2, reg_y1, reg_y2 = 150, 450, 515, 595
boarding_state = "Wait Boarding"
complete_time = 0
rect = Rectangle((reg_x1/100, 6-reg_y2/100), (reg_x2-reg_x1)/100, (reg_y2-reg_y1)/100, linewidth=1, edgecolor='g', facecolor='none')
ax2.add_patch(rect)

# Iterate over video frames
while videocapture.isOpened():
    
    def animate_door1(i):
        
        success, f = videocapture.read()
        
        if not success:
            return 1
        
        ax1_title = 'BEV boarding view: ' + str(round(videocapture.get(cv2.CAP_PROP_POS_MSEC)/1000, 1)) + ' sec'
        ax1.set_title(ax1_title)
        ax3.set_title('predict boarding passengers')
    
        ##### 1. initial setting #####
        ax2.cla()
        ax2.set_xlim(0, 6)
        ax2.set_ylim(-1, 5)
        rect = Rectangle((reg_x1/100, 6-reg_y2/100), (reg_x2-reg_x1)/100, (reg_y2-reg_y1)/100, linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        
        people_bottom = []

        # people = []
        people_ori = []
        
        colors = ['blue', 'green', 'purple', 'pink', 'orange', 'yellow']

        data = []
        ori = []
        cluster = []
        c_ori = []
        noise = []
        n_ori = []
        eps = 0.8 # 반지름 (epsilon)
        minPoints = 1
        new_ori = []
        people_center=[]
        
        in_region = False
        
        f2 = f.copy()
        
        
        ### IPM (BEV transform) ###
        # Run YOLOv8 inference on the frame
        results = model(f, classes=0, verbose=False)

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
                
                if y1 + (x2-x1)*2.3 > 600 or y2-y1 > (x2-x1)*2:
                    people_bottom.append([[int((x1+x2)/2), int(y2)]])
                else:
                    people_bottom.append([[int((x1+x2)/2), int(y1 + (x2-x1)*2.3)]])
                    

        frame = f[0:600, 60:600]
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

        ##### 2. start boarding #####
        
        global boarding_state
        global cnt
        global queue_cnt
        global complete_time
        global num
        global start_cnt
        
        cv2.rectangle(frame, (reg_x1, reg_y1), (reg_x2, reg_y2), (0, 255, 0), 2)
        cv2.rectangle(f, (reg_x1, reg_y1), (reg_x2, reg_y2), (0, 255, 0), 2)
        
        if boarding_state == "Wait Boarding":
            
            start_cnt += 1
            
            ax2.set_title(boarding_state)
                    
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
                    if 140 < person_ori < 220:
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
        
                boarding_state = "Start Boarding Passengers Recognition"
                ax2.set_title(boarding_state)
                print(boarding_state)


        ##### 3. boarding passengers recognition #####
        elif boarding_state == "Start Boarding Passengers Recognition":
            
            for person_bottom, person_ori in zip(transformed_people_bottom, people_ori):

                bottom_x, bottom_y = person_bottom

                # if 100 < person_ori < 260:
                if 140 < person_ori < 220:
                    ori.append(person_ori)
                    data.append([bottom_x, bottom_y])

                else:
                    bx = bottom_x/100
                    by = 6-bottom_y/100
                    p_ori = person_ori-270 if person_ori+90>=360 else person_ori+90
                    noise.append((bx, by)) # noise 리스트에 추가
                    n_ori.append(p_ori)
                    
            points = calculate_new_data(noise, n_ori, eps) 
            
            for i, v in enumerate(points): # 튜플의 형태로 noise의 길이만큼 반복
                end = calculate_new_data_reverse(v, n_ori[i], eps)
                # cv2.circle(frame, (int(end[0]*100), int((6 - end[1])*100)), 4, (0, 0, 0), -1)
                ax2.plot(end[0], end[1], marker='x', color='grey', linestyle='') # 그래프 초기화
                draw_arrow(end[0], end[1], n_ori[i], eps, ax2)
                # ax.plot(v[0], v[1], marker='x', color='grey', linestyle='', markersize=eps*4) # 그래프 초기화
            
            noise = []
            n_ori = []
            points = []
        
            data = [[x / 100, 6 - y / 100] for x, y in data]

            new_ori = calculate_new_ori(ori)

            A = np.array(data)
            B = np.array(new_ori)

            sorted_indices = np.argsort(-A[:, 1])
            data = A[sorted_indices].tolist()
            new_ori = B[sorted_indices].tolist()

            points = calculate_new_data(data, new_ori, eps) 
            
            # print(points)

            # print("Input Data : ", data) # 데이터 출력
            # print("Ori Data : ", ori)
            
            min_y_value = float(10)  # 초기 최소값으로 설정
            min_x_value = float(10)

            # print('-'*50) 
            cluster = cluster_new(data, points, eps) # points를 clustering 실행 후 idx로 초기화
            
            for c in cluster:
                if len(c)<=1:
                    cluster.remove(c)
                    noise.append(c[0])

            cnt = []
            queue = []
            cluster2 = [item for sublist in cluster for item in sublist]
            
            for index, coord in enumerate(cluster2):
                x, y = coord
                if y < min_y_value:
                    min_y_value = y
                    min_x_value = x

            # print(min_x_value, min_y_value)
            min_y_queue = 0

            for sublist in cluster:
                if (min_x_value, min_y_value) in sublist:
                    cnt = len(sublist)
                    queue = sublist
                    
                    break
                                    
                else:
                    cnt = 0                
            
            # print(data, cluster)
            
            t_people_bottom = [[x / 100, 6 - y / 100] for x, y in transformed_people_bottom]
            # print(t_people_bottom)
            
            for i, c in enumerate(cluster): # 튜플의 형태로 cluster의 길이만큼 반복
                for j, v in enumerate(c): # 튜플의 형태로 c의 길이만큼 반복
                    
                    if v == (min_x_value, min_y_value):
                        min_y_queue = v[1]
                    
                    idx = data.index(list(v))
                    idx2 = t_people_bottom.index([v[0], v[1]])
                    
                    draw_arrow(v[0], v[1], new_ori[idx], eps, ax2)
                    
                    if c == queue:
                        
                        complete_time = 0
                        
                        draw_eps(points[idx][0], points[idx][1], eps*2, 'red', ax2)
                        ax2.plot(v[0], v[1], marker='o', color='red', linestyle='') # 그래프 초기화
                        cv2.circle(f2, (int(people_center[idx2][0]), int((people_center[idx2][1]))), 4, (0, 0, 255), -1)
                    else:
                        draw_eps(points[idx][0], points[idx][1], eps*2, colors[i], ax2)
                        ax2.plot(v[0], v[1], marker='.', color=colors[i], linestyle='') # 그래프 초기화
                        # cv2.circle(frame, (int(end[0]*100), int((6 - end[1])*100)), 4, (0, 0, 0), -1)


            for i, v in enumerate(noise): # 튜플의 형태로 noise의 길이만큼 반복
                
                idx = data.index(list(v))
                
                ax2.plot(v[0], v[1], marker='x', color='grey', linestyle='') # 그래프 초기화
                # cv2.circle(frame, (int(end[0]*100), int((6 - end[1])*100)), 4, (0, 0, 0), -1)
                draw_arrow(v[0], v[1], new_ori[idx], eps, ax2)
                # ax.plot(v[0], v[1], marker='x', color='grey', linestyle='', markersize=eps*4) # 그래프 초기화

            print("Boarding & Queue Counting")
            # print(min_y_queue)

            if cnt != 0:
                # print('queue counting: '+ str(cnt))
                ax2.set_title('queue counting: '+ str(cnt))
            else:
                # print('**queue counting: 0**')
                ax2.set_title('queue counting: 0')
        
            if min_y_queue >= 3 or cnt <= 1:
                
                queue_cnt += 1
                
                if queue_cnt == 3:
            
                    boarding_state = "All Passengers in Queue have boarded"
                    ax2.set_title(boarding_state)
                    print(boarding_state)
                
            else:
                queue_cnt = 0
        
        
        elif boarding_state == "All Passengers in Queue have boarded":
        
            
            ax2.set_title("Check Boarding Complete")
            
            if len(transformed_people_bottom)>0:
            
                for i, p in enumerate(transformed_people_bottom):
                    
                    bottom_x, bottom_y = p
                    bx = bottom_x/100
                    by = 6-bottom_y/100

                    p_ori = people_ori[i]-270 if people_ori[i]+90>=360 else people_ori[i]+90
                    
                    if reg_x1 < bottom_x < reg_x2 and reg_y1 < bottom_y < reg_y2:
                        
                        draw_arrow(bx, by, p_ori, eps, ax2)
                        
                        # if 100 < people_ori[i] < 260:
                        if 140 < people_ori[i] < 220:
                            
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
                print("Keep Boarding, Reset Boarding Complete Count")
                complete_time = 0
                
            elif in_region == False:
                
                complete_time += 1
                
                if complete_time%15 == 0:
                    print("Boarding Complete Count: " + str(complete_time//15) + " sec")

                
                elif complete_time == 48: # fps = 15/1
                    print("Boarding Complete Count: 3.2 sec, " + boarding_state)
                    boarding_state = "Boarding Complete"    
                    ax2.set_title(boarding_state)
                    # print(boarding_state)
                    

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        ax1.imshow(frame)
        ax3.imshow(f2)
        # print(boarding_state)
        # ax2.set_title(boarding_state)
        
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
        
        if boarding_state == "Boarding Complete":
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