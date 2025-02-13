import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import numpy as np
import random
from matplotlib import style
import cv2
from tools.hboe import hboe

def return_bbox(results, frame):

    people = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            b = box.xyxy[0]
            x1, y1, x2, y2 = b.tolist()

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            people.append([x, y])

            frame = cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return people, frame

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

def draw_arrow(x, y, angle_deg, length, ax):
    angle_rad = np.deg2rad(angle_deg)
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    ax.arrow(x, y, dx, dy, head_width=0.07, head_length=0.07, fc='black', ec='black')

def draw_eps(x, y, eps, colors, ax):
    # 플롯 생성 (마커는 'o'로 설정하고, 투명도는 0.5, 원의 반지름은 eps으로 설정)
    ax.plot(x, y, marker='o', linestyle='', alpha=0.15, color=colors, markersize=eps*50)

def Euclidean(v1, v2): # 유클리드 거리 계산 함수
    sum = 0
    for i in range(len(v1)):
        sum +=  (v1[i]-v2[i])**2
    return round(math.sqrt(sum), 2)

def search(r, eps): # 근접한 클러스터 찾는 함수
    neighborP = [] # 근접 클러스터를 담을 리스트
    for i in range(len(r)): # 리스트 r의 길이만큼 반복
        if r[i] < eps: # 만약 r 리스트의 i번째 값이 eps 보다 작으면
            neighborP.append(i) # 근접 클러스터에 추가
    return neighborP # neighborP 리턴

def expansion(i, idx, dist, neighborP, visited,  C, eps): # 클러스터 통합하는 함수
    idx[i] = C # C값 idx 리스트에 저장 (마이크로 클러스터)
    k = 0 # k값 0으로 초기화
    while len(neighborP) > k: # 이웃리스트 길이가 k보다 큰 경우
        j = neighborP[k] # 이웃리스트의 k번째 값을 j에 저장
        if(visited[j]) == False: # 방문 False인 경우
            visited[j] = True # True로 초기화
            neighbor_of_neighbor = search(dist[j], eps) # 반경 안에 있는 점들을 저장
            if len(neighbor_of_neighbor) > 2: # 만약 neighbor_of_neighbor의 길이가 2보다 큰 경우
                neighborP += neighbor_of_neighbor # neighborP에 추가  
        if idx[j] == 0: # idx의 j번째 값이 0인 경우
            idx[j] = C # c로 초기화
        k+=1 # 계산 종료되면 K+1

def cluster_new(points, data, eps, minPoints): # New Clustering 방식
    n = len(points) # points(입력받는 data)의 길이 n에 저장
    dist = [[0] * n for _ in range(n)] # n의 길이만큼 이중리스트 dist 생성
    for i in range(n): # 행의 길이만큼 반복
        for j in range(n): # 열의 길이만큼 반복
            if i == j: # 만약 i==j인 경우 
                continue # continue
            else: # 그 외의 경우
                # dist[i][j] = Euclidean(points[i], points[j]) # 거리를 구하고 해당 행열에 추가
                dist[i][j] = Euclidean(points[i], data[j])
    visited = [False for _ in range(n)] # n의 길이만큼 boolean형 리스트 생성
    noise = [False for _ in range(n)] # n의 길이만큼 boolean형 리스트 생성
    idx = [0 for _ in range(n)] # n의 길이만큼 리스트 생성

    C = 0 # c값 0으로 초기화

    for i, point in enumerate(points): # 튜플의 형태로 points의 길이만큼 반복
        if visited[i] == False: # 만약 i번째에 방문하지 않았다면
            visited[i] = True # True로 바꿈
            neighborP = search(dist[i], eps) # 클러스터가 해당 반경 내에 있으면 클러스터에 추가
            if len(neighborP) > minPoints: # 만약 neighborP가 minPoints보다 클 경우
                C += 1 # C의 값이 1 증가
                expansion(i, idx, dist, neighborP, visited, C, eps) # 마이크로 클러스터 확장
            else: # 그 외의 경우
                noise[i] = True # noise 값을 True로 변경
    return idx # idx 리턴

class QueueDetection(Node):
    def __init__(self):
        super().__init__('queue_detection')

        self.subscription_door1 = self.create_subscription(
            Image,
            'door1_out',
            self.door1_callback,
            1)
        self.subscription_door1

        self.subscription_door2 = self.create_subscription(
            Image,
            'door2_out',
            self.door2_callback,
            1)
        self.subscription_door2
        '''
        self.subscription_door3 = self.create_subscription(
            Image,
            'door3_out',
            self.door3_callback,
            1)
        self.subscription_door3
        '''
        
        self.bridge = CvBridge()

        self.model = YOLO('yolov8x.pt')

        self.people_door1 = []
        self.people_door2 = []
        '''
        self.people_door3 = []

        '''
        self.frame_door1 = None
        self.frame_door2 = None
        '''
        self.frame_door3 = None
        '''


        self.fig = plt.figure(figsize = (14, 7))

        self.ax1 = plt.subplot(1,2,1)
        self.ax1.set_title('original')
        self.ax1.axis("off")

        self.ax2 = plt.subplot(1, 2, 2)
        self.ax2.set_xlim(-1, 6)
        self.ax2.set_ylim(-1, 6)

        self.timer_door1 = self.create_timer(0.02, self.queue_detection_pub_door1)
        self.timer_door2 = self.create_timer(0.02, self.queue_detection_pub_door2)
        '''
        self.timer_door3 = self.create_timer(0.02, self.queue_detection_pub_door3)
        '''
        
        self.publisher_door1_detect = self.create_publisher(Image,'detect_door1_out', 1)
        self.publisher_door2_detect = self.create_publisher(Image,'detect_door2_out', 1)
        '''
        self.publisher_door3_detect = self.create_publisher(Image,'detect_door3_in', 1)
        '''

        self.publisher_door1_queue = self.create_publisher(Image,'door1_queue', 1)
        self.publisher_door2_queue = self.create_publisher(Image,'door2_queue', 1)
        '''
        self.publisher_door3_queue = self.create_publisher(Image,'door3_queue', 1)
        '''

    def door1_callback(self, msg):

        self.frame_door1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_door1, classes=0)

        self.people_door1 = return_bbox(results, self.frame_door1)[0]
        self.frame_door1 = return_bbox(results, self.frame_door1)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door1, encoding='bgr8')

        self.publisher_door1_detect.publish(detection_msg)

    def door2_callback(self, msg):

        self.frame_door2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_door2, classes=0)

        self.people_door2 = return_bbox(results, self.frame_door2)[0]
        self.frame_door2 = return_bbox(results, self.frame_door2)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door2, encoding='bgr8')

        self.publisher_door2_detect.publish(detection_msg)
    
    '''
    def door3_callback(self, msg):

        self.frame_door3 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_door3, classes=0)

        self.people_door3 = return_bbox(results, self.frame_door3)[0]
        self.frame_door3 = return_bbox(results, self.frame_door3)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door3, encoding='bgr8')

        self.publisher_door3_detect.publish(detection_msg)
    '''

# 클러스터링 함수 하나로...만들어야긋다....겁내 기네
    def queue_detection_pub_door1(self):

        fig = self.fig
        ax1 = self.ax1
        ax2 = self.ax2

        def animate_door1(i):

            plt.cla()
            ax2.set_xlim(-1, 6)
            ax2.set_ylim(-1, 6)

            colors = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'yellow']

            people = self.people_door1 # human 좌표
            data = []
            ori = []
            cluster = []
            c_ori = []
            noise = []
            n_ori = []
            eps = 1 # 반지름 (epsilon)
            minPoints = 1
            new_ori = []

            for person in people:

                x1, y1, x2, y2 = person

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # HBOE를 위해 사람 단위 별로 crop
                crop_person = frame[y1:y2, x1:x2]

                # Human Body Orientation Recognition 진행 -> 각도 출력
                person_ori = int(hboe(crop_person))

                if 100 < person_ori < 260:
                    ori.append(person_ori)
                    data.append([center_x, center_y])

                else:
                    cx = center_x/100
                    cy = 6-center_y/100
                    p_ori = person_ori-270 if person_ori+90>=360 else person_ori+90
                    ax2.plot(cx, cy, marker='x', color='grey', linestyle='') # 그래프 초기화
                    draw_arrow(cx, cy, p_ori, eps, ax2)


            data = [[x / 100, 6 - y / 100] for x, y in data]

            new_ori = calculate_new_ori(ori)

            A = np.array(data)
            B = np.array(new_ori)

            sorted_indices = np.argsort(-A[:, 1])
            data = A[sorted_indices].tolist()
            new_ori = B[sorted_indices].tolist()

            points = calculate_new_data(data, new_ori, eps) 

            print("Input Data : ", data) # 데이터 출력
            print("Ori Data : ", ori)

            print('-'*50) 
            idx = cluster_new(points, data, eps*2, minPoints) # points를 clustering 실행 후 idx로 초기화

            for i in range(max(idx)): # idx의 최대값만큼 반복
                k = [] # k 리스트 초기화
                for j in range(len(idx)): # idx의 길이만큼 반복
                    if idx[j] == i+1: # idx의 j번째 값이 i+1과 같으면
                        k.append(j) # k 리스트에 추가
                result = [] # result 리스트 초기화
                o = []
                for p in range(len(k)): # k 리스트의 길이만큼 반복
                    result.append(points[k[p]]) # result 리스트에 points의 k[p]번째 값 추가
                    o.append(new_ori[k[p]])
                cluster.append(result) # cluster에 result 추가
                c_ori.append(o)

            for i in range(len(idx)): # DBSCAN을 통해 구한 idx의 길이만큼 반복
                if idx[i] == 0: # 만약 idx의 i번째 값이 0인 경우
                    noise.append(points[i]) # noise 리스트에 추가
                    n_ori.append(new_ori[i])

            cnt = []

            for i, c in enumerate(cluster): # 튜플의 형태로 cluster의 길이만큼 반복
                print('Cluster ', i , ': ', c) # 클러스터 출력
                cnt.append(len(c))
            print('-'*50)
            print('noise : ', noise) # 노이즈 출력

            for i, c in enumerate(cluster): # 튜플의 형태로 cluster의 길이만큼 반복
                for j, v in enumerate(c): # 튜플의 형태로 c의 길이만큼 반복
                    end = calculate_new_data_reverse(v, c_ori[i][j], eps)
                    ax2.plot(end[0], end[1], marker='o', color=colors[i], linestyle='') # 그래프 초기화
                    draw_eps(v[0], v[1], eps*2, colors[i], ax2)
                    draw_arrow(end[0], end[1], c_ori[i][j], eps, ax2)
                    # ax.plot(v[0], v[1], marker='o', color=colors[i], linestyle='', markersize=eps*4) # 그래프 초기화


            for i, v in enumerate(noise): # 튜플의 형태로 noise의 길이만큼 반복
                end = calculate_new_data_reverse(v, n_ori[i], eps)
                ax2.plot(end[0], end[1], marker='x', color='grey', linestyle='') # 그래프 초기화
                draw_arrow(end[0], end[1], n_ori[i], eps, ax2)
                # ax.plot(v[0], v[1], marker='x', color='grey', linestyle='', markersize=eps*4) # 그래프 초기화


            # plt.title('New Clustering Result', fontsize=10) # 타이틀 DBSCAN 설정, 폰트사이즈 10
            # ax2.xlabel('X', fontsize=10) # X 라벨값 부여, 폰트사이즈 10
            # ax2.ylabel('Y', fontsize=10) # Y 라벨값 부여, 폰트사이즈 10
            if len(cnt)!=0:
                ax2_title = 'queue counting: '+ str(max(cnt))
            else:
                ax2_title = 'queue counting: 0'
            ax2.set_title(ax2_title)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax1.imshow(frame)

        ani = FuncAnimation(plt.gcf(), animate_door1, interval=1000)
        plt.tight_layout()

        # 이거 되나?
        queue_detection = self.bridge.cv2_to_imgmsg(plt, encoding='bgr8')
        self.publisher_door1_queue.publish(queue_detection)

    def queue_detection_pub_door2(self):
        
        fig = self.fig
        ax1 = self.ax1
        ax2 = self.ax2

        def animate_door2(i):

            plt.cla()
            ax2.set_xlim(-1, 6)
            ax2.set_ylim(-1, 6)

            colors = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'yellow']

            people = self.people_door2 # human 좌표
            data = []
            ori = []
            cluster = []
            c_ori = []
            noise = []
            n_ori = []
            eps = 1 # 반지름 (epsilon)
            minPoints = 1
            new_ori = []

            for person in people:

                x1, y1, x2, y2 = person

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # HBOE를 위해 사람 단위 별로 crop
                crop_person = frame[y1:y2, x1:x2]

                # Human Body Orientation Recognition 진행 -> 각도 출력
                person_ori = int(hboe(crop_person))

                if 100 < person_ori < 260:
                    ori.append(person_ori)
                    data.append([center_x, center_y])

                else:
                    cx = center_x/100
                    cy = 6-center_y/100
                    p_ori = person_ori-270 if person_ori+90>=360 else person_ori+90
                    ax2.plot(cx, cy, marker='x', color='grey', linestyle='') # 그래프 초기화
                    draw_arrow(cx, cy, p_ori, eps, ax2)


            data = [[x / 100, 6 - y / 100] for x, y in data]

            new_ori = calculate_new_ori(ori)

            A = np.array(data)
            B = np.array(new_ori)

            sorted_indices = np.argsort(-A[:, 1])
            data = A[sorted_indices].tolist()
            new_ori = B[sorted_indices].tolist()

            points = calculate_new_data(data, new_ori, eps) 

            # print("Input Data : ", data) # 데이터 출력
            # print("Ori Data : ", ori)

            # print('-'*50) 
            idx = cluster_new(points, data, eps*2, minPoints) # points를 clustering 실행 후 idx로 초기화

            for i in range(max(idx)): # idx의 최대값만큼 반복
                k = [] # k 리스트 초기화
                for j in range(len(idx)): # idx의 길이만큼 반복
                    if idx[j] == i+1: # idx의 j번째 값이 i+1과 같으면
                        k.append(j) # k 리스트에 추가
                result = [] # result 리스트 초기화
                o = []
                for p in range(len(k)): # k 리스트의 길이만큼 반복
                    result.append(points[k[p]]) # result 리스트에 points의 k[p]번째 값 추가
                    o.append(new_ori[k[p]])
                cluster.append(result) # cluster에 result 추가
                c_ori.append(o)

            for i in range(len(idx)): # DBSCAN을 통해 구한 idx의 길이만큼 반복
                if idx[i] == 0: # 만약 idx의 i번째 값이 0인 경우
                    noise.append(points[i]) # noise 리스트에 추가
                    n_ori.append(new_ori[i])

            cnt = []

            for i, c in enumerate(cluster): # 튜플의 형태로 cluster의 길이만큼 반복
                # print('Cluster ', i , ': ', c) # 클러스터 출력
                cnt.append(len(c))
            # print('-'*50)
            # print('noise : ', noise) # 노이즈 출력

            for i, c in enumerate(cluster): # 튜플의 형태로 cluster의 길이만큼 반복
                for j, v in enumerate(c): # 튜플의 형태로 c의 길이만큼 반복
                    end = calculate_new_data_reverse(v, c_ori[i][j], eps)
                    ax2.plot(end[0], end[1], marker='o', color=colors[i], linestyle='') # 그래프 초기화
                    draw_eps(v[0], v[1], eps*2, colors[i], ax2)
                    draw_arrow(end[0], end[1], c_ori[i][j], eps, ax2)
                    # ax.plot(v[0], v[1], marker='o', color=colors[i], linestyle='', markersize=eps*4) # 그래프 초기화


            for i, v in enumerate(noise): # 튜플의 형태로 noise의 길이만큼 반복
                end = calculate_new_data_reverse(v, n_ori[i], eps)
                ax2.plot(end[0], end[1], marker='x', color='grey', linestyle='') # 그래프 초기화
                draw_arrow(end[0], end[1], n_ori[i], eps, ax2)
                # ax.plot(v[0], v[1], marker='x', color='grey', linestyle='', markersize=eps*4) # 그래프 초기화


            # plt.title('New Clustering Result', fontsize=10) # 타이틀 DBSCAN 설정, 폰트사이즈 10
            # ax2.xlabel('X', fontsize=10) # X 라벨값 부여, 폰트사이즈 10
            # ax2.ylabel('Y', fontsize=10) # Y 라벨값 부여, 폰트사이즈 10
            if len(cnt)!=0:
                ax2_title = 'queue counting: '+ str(max(cnt))
            else:
                ax2_title = 'queue counting: 0'
            ax2.set_title(ax2_title)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax1.imshow(frame)

        ani = FuncAnimation(plt.gcf(), animate_door2, interval=1000)
        plt.tight_layout()

        # Convert the plot to an image
        fig.canvas.draw()
        queue_result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        queue_result = queue_result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # 이거 되나?
        queue_detection = self.bridge.cv2_to_imgmsg(queue_result, encoding='bgr8')
        self.publisher_door2_queue.publish(queue_detection)

    '''
    def queue_detection_pub_door3(self):

        def animate_door3(i):

            ax1 = self.ax1
            ax2 = self.ax2

            plt.cla()
            ax2.set_xlim(-1, 6)
            ax2.set_ylim(-1, 6)

            colors = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'yellow']

            people = self.people_door3 # human 좌표
            data = []
            ori = []
            cluster = []
            c_ori = []
            noise = []
            n_ori = []
            eps = 1 # 반지름 (epsilon)
            minPoints = 1
            new_ori = []

            for person in people:

                x1, y1, x2, y2 = person

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # HBOE를 위해 사람 단위 별로 crop
                crop_person = frame[y1:y2, x1:x2]

                # Human Body Orientation Recognition 진행 -> 각도 출력
                person_ori = int(hboe(crop_person))

                if 100 < person_ori < 260:
                    ori.append(person_ori)
                    data.append([center_x, center_y])

                else:
                    cx = center_x/100
                    cy = 6-center_y/100
                    p_ori = person_ori-270 if person_ori+90>=360 else person_ori+90
                    ax2.plot(cx, cy, marker='x', color='grey', linestyle='') # 그래프 초기화
                    draw_arrow(cx, cy, p_ori, eps, ax2)


            data = [[x / 100, 6 - y / 100] for x, y in data]

            new_ori = calculate_new_ori(ori)

            A = np.array(data)
            B = np.array(new_ori)

            sorted_indices = np.argsort(-A[:, 1])
            data = A[sorted_indices].tolist()
            new_ori = B[sorted_indices].tolist()

            points = calculate_new_data(data, new_ori, eps) 

            print("Input Data : ", data) # 데이터 출력
            print("Ori Data : ", ori)

            print('-'*50) 
            idx = cluster_new(points, data, eps*2, minPoints) # points를 clustering 실행 후 idx로 초기화

            for i in range(max(idx)): # idx의 최대값만큼 반복
                k = [] # k 리스트 초기화
                for j in range(len(idx)): # idx의 길이만큼 반복
                    if idx[j] == i+1: # idx의 j번째 값이 i+1과 같으면
                        k.append(j) # k 리스트에 추가
                result = [] # result 리스트 초기화
                o = []
                for p in range(len(k)): # k 리스트의 길이만큼 반복
                    result.append(points[k[p]]) # result 리스트에 points의 k[p]번째 값 추가
                    o.append(new_ori[k[p]])
                cluster.append(result) # cluster에 result 추가
                c_ori.append(o)

            for i in range(len(idx)): # DBSCAN을 통해 구한 idx의 길이만큼 반복
                if idx[i] == 0: # 만약 idx의 i번째 값이 0인 경우
                    noise.append(points[i]) # noise 리스트에 추가
                    n_ori.append(new_ori[i])

            cnt = []

            for i, c in enumerate(cluster): # 튜플의 형태로 cluster의 길이만큼 반복
                print('Cluster ', i , ': ', c) # 클러스터 출력
                cnt.append(len(c))
            print('-'*50)
            print('noise : ', noise) # 노이즈 출력

            for i, c in enumerate(cluster): # 튜플의 형태로 cluster의 길이만큼 반복
                for j, v in enumerate(c): # 튜플의 형태로 c의 길이만큼 반복
                    end = calculate_new_data_reverse(v, c_ori[i][j], eps)
                    ax2.plot(end[0], end[1], marker='o', color=colors[i], linestyle='') # 그래프 초기화
                    draw_eps(v[0], v[1], eps*2, colors[i], ax2)
                    draw_arrow(end[0], end[1], c_ori[i][j], eps, ax2)
                    # ax.plot(v[0], v[1], marker='o', color=colors[i], linestyle='', markersize=eps*4) # 그래프 초기화


            for i, v in enumerate(noise): # 튜플의 형태로 noise의 길이만큼 반복
                end = calculate_new_data_reverse(v, n_ori[i], eps)
                ax2.plot(end[0], end[1], marker='x', color='grey', linestyle='') # 그래프 초기화
                draw_arrow(end[0], end[1], n_ori[i], eps, ax2)
                # ax.plot(v[0], v[1], marker='x', color='grey', linestyle='', markersize=eps*4) # 그래프 초기화


            # plt.title('New Clustering Result', fontsize=10) # 타이틀 DBSCAN 설정, 폰트사이즈 10
            # ax2.xlabel('X', fontsize=10) # X 라벨값 부여, 폰트사이즈 10
            # ax2.ylabel('Y', fontsize=10) # Y 라벨값 부여, 폰트사이즈 10
            if len(cnt)!=0:
                ax2_title = 'queue counting: '+ str(max(cnt))
            else:
                ax2_title = 'queue counting: 0'
            ax2.set_title(ax2_title)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax1.imshow(frame)

        ani = FuncAnimation(plt.gcf(), animate_door3, interval=1000)
        plt.tight_layout()

        # 이거 되나?
        queue_detection = self.bridge.cv2_to_imgmsg(plt, encoding='bgr8')
        self.publisher_door3_queue.publish(queue_detection)
    '''

def main(args=None):
    rclpy.init(args=args)

    node = QueueDetection()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()